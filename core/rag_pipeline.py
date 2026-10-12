"""
RAG Pipeline — Document Intelligence
Stack : LangChain + Groq AI + ChromaDB + HuggingFace

Architecture :
  Document(s) → Chunking → Embeddings (HuggingFace) → ChromaDB
                                                        ↓
  Question → Embedding → Retrieval (top-k chunks) → LLM (Groq) → Réponse
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ── Prompt système ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un assistant expert en analyse de documents techniques et scientifiques.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Cite les sources avec leur nom et leur page lorsqu'ils sont disponibles.
Rédige une réponse complète et naturelle dans la même langue que la question.
Ne recopie pas des fragments bruts et ne coupe pas une phrase au milieu.
Structure la réponse avec une réponse directe, puis les éléments utiles et leurs sources.
Si les passages sont insuffisants ou contradictoires, explique clairement cette limite.

Contexte :
{context}

Question : {question}

Réponse :"""
)

SUMMARY_PROMPT = """Analyse ce document et produis un résumé structuré en JSON avec :
- "title": titre exact s'il est visible, sinon sujet principal
- "summary": synthèse exécutive claire en 4 à 6 phrases, sans répéter le titre
- "objectives": objectif ou problématique du document
- "methodology": méthodes, données et protocole décrits
- "key_findings": résultats majeurs, avec chiffres uniquement s'ils figurent dans le texte
- "conclusion": conclusion et recommandations du document
- "limitations": limites reconnues par les auteurs, ou "Non précisées dans le document"
- "key_points": liste de 3 à 5 informations distinctes, sans répétition
- "domain": domaine principal du document
- "language": langue principalement utilisée dans le document
- "complexity": Débutant, Intermédiaire ou Expert

Règles : écris dans la langue principale du document. N'invente aucune information.
Distingue explicitement les résultats des auteurs des comparaisons à la littérature.
Le texte fourni contient des extraits répartis dans le document : synthétise-les ensemble.

Extraits du document :
{text}

Retourne uniquement un objet JSON valide, sans balises Markdown ni texte avant/après."""


def select_summary_excerpt(full_text: str, max_chars: int = 16000, windows: int = 8) -> str:
    """Sample evenly across long documents so the summary is not introduction-only."""
    text = full_text.strip()
    if len(text) <= max_chars:
        return text

    window_count = max(1, min(windows, max_chars // 1000))
    label_budget = window_count * 24
    window_budget = max(1, (max_chars - label_budget) // window_count)
    window_size = max(1, window_budget - 500)
    last_start = len(text) - window_size
    starts = []
    for index in range(window_count):
        target = round(index * last_start / max(window_count - 1, 1))
        line_start = text.rfind("\n", 0, target) + 1
        if target - line_start <= window_size * 0.5:
            target = line_start
        starts.append(target)
    starts = list(dict.fromkeys(starts))
    excerpts = []
    for index, start in enumerate(starts):
        if start > 0 and index == window_count - 1:
            previous_boundary = text.rfind("\n", 0, start)
            if previous_boundary >= 0:
                start = previous_boundary + 1
        elif start > 0:
            next_boundary = min(
                (
                    position + len(separator)
                    for separator in ("\n", ". ", "? ", "! ")
                    if (position := text.find(separator, start, min(start + 250, len(text)))) >= 0
                ),
                default=start,
            )
            start = next_boundary
        end = min(start + window_size, len(text))
        if end < len(text):
            candidates = [
                position
                for separator in ("\n", ". ", "? ", "! ")
                if (position := text.find(separator, end, min(end + 500, len(text)))) >= 0
            ]
            if candidates:
                end = min(candidates) + 1
        excerpts.append(f"[Extrait {index + 1}/{window_count}]\n{text[start:end].strip()}")
    return "\n\n".join(excerpts)


def expand_retrieved_chunks(vectorstore, relevant_docs: list, neighbors: int = 1) -> list:
    """Add adjacent chunks to each match so the model receives complete context."""
    by_source: dict[str, dict[int, Document]] = {}
    selected: dict[str, set[int]] = {}
    for document in relevant_docs:
        source = str(document.metadata.get("source", "unknown"))
        chunk_id = document.metadata.get("chunk_id")
        if isinstance(chunk_id, int):
            selected.setdefault(source, set()).update(
                range(max(0, chunk_id - neighbors), chunk_id + neighbors + 1)
            )
        else:
            by_source.setdefault(source, {})[-1] = document

    for source in selected:
        stored = vectorstore.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        source_chunks = by_source.setdefault(source, {})
        for metadata, content in zip(stored["metadatas"], stored["documents"]):
            chunk_id = metadata.get("chunk_id")
            if isinstance(chunk_id, int) and chunk_id in selected[source]:
                source_chunks[chunk_id] = Document(
                    page_content=content,
                    metadata=metadata,
                )

    return [
        document
        for source in sorted(by_source)
        for _, document in sorted(by_source[source].items())
    ]


def merge_chunk_overlaps(chunks: list[Document]) -> str:
    """Join adjacent chunks without duplicating their configured overlap text."""
    merged = ""
    for chunk in chunks:
        content = chunk.page_content.strip()
        overlap_limit = min(len(merged), len(content), 500)
        overlap = max(
            (
                size
                for size in range(1, overlap_limit + 1)
                if merged.endswith(content[:size])
            ),
            default=0,
        )
        if merged and content:
            merged += content[overlap:] if overlap else "\n\n" + content
        else:
            merged += content
    return merged


def group_chunks_by_source_page(chunks: list[Document]) -> list[tuple[tuple[str, str], list[Document]]]:
    ordered = sorted(
        chunks,
        key=lambda item: (
            str(item.metadata.get("source", "unknown")),
            str(item.metadata.get("page", "non précisée")),
            item.metadata.get("chunk_id", -1),
        ),
    )
    groups = []
    current_key = None
    current_chunks = []
    previous_chunk_id = None
    for document in ordered:
        source = str(document.metadata.get("source", "unknown"))
        page = document.metadata.get("page")
        page_label = str(page) if page is not None else "non précisée"
        key = (source, page_label)
        chunk_id = document.metadata.get("chunk_id")
        consecutive = (
            current_key == key
            and isinstance(chunk_id, int)
            and isinstance(previous_chunk_id, int)
            and chunk_id == previous_chunk_id + 1
        )
        if current_chunks and not consecutive:
            groups.append((current_key, current_chunks))
            current_chunks = []
        if not current_chunks:
            current_key = key
        current_chunks.append(document)
        previous_chunk_id = chunk_id
    if current_chunks:
        groups.append((current_key, current_chunks))
    return groups


class RAGPipeline:
    """
    Pipeline RAG complet :
    - Ingestion de documents (PDF, DOCX, TXT)
    - Chunking intelligent avec RecursiveTextSplitter
    - Embeddings via HuggingFace (all-MiniLM-L6-v2)
    - Stockage vectoriel ChromaDB (persistant)
    - Retrieval + génération avec Groq
    """

    def __init__(
        self,
        api_key: str,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 4,
        temperature: float = 0.1,
    ):
        os.environ["GROQ_API_KEY"] = api_key
        self.api_key = api_key
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.top_k = top_k
        self.temperature = temperature

        # Embeddings HuggingFace (local/free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # LLM Groq
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=api_key,
            max_tokens=2048,
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # Vector store
        self.vectorstore: Optional[Chroma] = None
        self.ingested_docs: List[Dict] = []
        self._load_existing_store()

    def _load_existing_store(self):
        """Charge le vectorstore existant s'il y en a un."""
        if os.path.exists(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
            except Exception:
                self.vectorstore = None

    def _doc_id(self, filename: str, content: bytes) -> str:
        """Return a stable ID for a filename and its contents."""
        return hashlib.sha256(filename.encode() + b"\0" + content).hexdigest()

    def load_document(self, file_path: str) -> List[Document]:
        """Charge un document selon son extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Format non supporté : {ext}. Formats acceptés : PDF, DOCX, TXT")
        return loader.load()

    def ingest(
        self,
        file_path: str,
        file_bytes: bytes = None,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingère un document dans le vectorstore.
        Retourne des stats sur l'ingestion.
        """
        filename = source_name or Path(file_path).name
        content = file_bytes if file_bytes is not None else Path(file_path).read_bytes()
        document_id = self._doc_id(filename, content)
        docs = self.load_document(file_path)

        # Chunking
        chunks = self.splitter.split_documents(docs)
        n_chunks = len(chunks)

        # Ajout de métadonnées
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source': filename,
                'chunk_id': i,
                'total_chunks': n_chunks,
                'document_id': document_id,
            })

        # Ajout au vectorstore
        chunk_ids = [f"{document_id}:{i}" for i in range(n_chunks)]
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.persist_dir,
                ids=chunk_ids,
            )
        else:
            self.vectorstore.delete(where={'source': filename})
            self.vectorstore.add_documents(chunks, ids=chunk_ids)

        stats = {
            'filename': filename,
            'n_pages': len(docs),
            'n_chunks': n_chunks,
            'avg_chunk_size': int(sum(len(c.page_content) for c in chunks) / max(n_chunks, 1)),
            'total_chars': sum(len(c.page_content) for c in chunks),
        }
        self.ingested_docs = [
            doc for doc in self.ingested_docs if doc['filename'] != filename
        ]
        self.ingested_docs.append(stats)
        return stats

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the configured chat model and return its response text."""
        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Pose une question sur les documents ingérés.
        Retourne la réponse + les sources + les chunks utilisés.
        """
        if self.vectorstore is None:
            raise ValueError("Aucun document ingéré. Charger des documents d'abord.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        # Récupérer les chunks pertinents
        relevant_docs = retriever.invoke(question)
        context_docs = expand_retrieved_chunks(self.vectorstore, relevant_docs, neighbors=1)
        context_groups = group_chunks_by_source_page(context_docs)

        # Construire le contexte
        context = "\n\n---\n\n".join(
            f"[Source : {source}; page : {page}]\n{merge_chunk_overlaps(chunks)}"
            for (source, page), chunks in context_groups
        )

        # Générer la réponse
        prompt = RAG_PROMPT.format(context=context, question=question)
        generation_error = None
        try:
            answer = self._invoke_llm(prompt)
        except Exception as e:
            generation_error = e
            answer = None

        result = {
            'question': question,
            'answer': answer,
            'n_sources': len(relevant_docs),
        }
        if generation_error is not None:
            result['generation_failed'] = True
            result['generation_error_type'] = type(generation_error).__name__
            status_code = getattr(generation_error, 'status_code', None)
            if isinstance(status_code, int):
                result['generation_status_code'] = status_code

        if return_sources:
            result['sources'] = []
            for (source, page_label), group in context_groups:
                first_chunk = group[0].metadata.get("chunk_id", 0)
                last_chunk = group[-1].metadata.get("chunk_id", first_chunk)
                result['sources'].append({
                    'source': source,
                    'chunk_id': first_chunk,
                    'first_chunk_id': first_chunk,
                    'last_chunk_id': last_chunk,
                    'page': page_label if page_label != "non précisée" else None,
                    'content': merge_chunk_overlaps(group),
                    'relevance_score': None,
                })

        return result

    def summarize(self, file_path: str = None, text: str = None) -> Dict[str, Any]:
        """
        Génère un résumé structuré d'un document.
        Peut prendre un fichier ou un texte brut.
        """
        if text is not None:
            full_text = text
        elif file_path:
            docs = self.load_document(file_path)
            full_text = "\n".join([d.page_content for d in docs])
        else:
            raise ValueError("Fournir file_path ou text")

        if not full_text.strip():
            raise ValueError("Le document ne contient pas de texte exploitable.")

        excerpt = select_summary_excerpt(full_text)
        prompt = SUMMARY_PROMPT.format(text=excerpt)
        import json
        try:
            content = self._invoke_llm(prompt).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            summary_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError("Le modèle a renvoyé un résumé non structuré. Réessayez.") from e

        if not isinstance(summary_data, dict):
            raise ValueError("Le modèle n’a pas renvoyé un objet de résumé valide.")
        required_fields = ("title", "summary", "objectives", "methodology",
                           "key_findings", "conclusion", "limitations", "key_points")
        missing_fields = [field for field in required_fields if not summary_data.get(field)]
        if missing_fields:
            raise ValueError(
                "Le résumé généré est incomplet : " + ", ".join(missing_fields)
            )
        if not isinstance(summary_data["key_points"], list):
            raise ValueError("Le modèle doit renvoyer les points clés sous forme de liste.")
        summary_data['full_text_length'] = len(full_text)
        summary_data['n_words'] = len(full_text.split())
        summary_data['sampled_text_length'] = len(excerpt)
        return summary_data

    def compare_documents(self, question: str) -> Dict[str, Any]:
        """
        Compare les réponses des différents documents ingérés
        sur une même question.
        """
        if not self.ingested_docs:
            raise ValueError("Aucun document ingéré.")

        results = {}
        for doc_info in self.ingested_docs:
            source = doc_info['filename']
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 2,
                    "filter": {"source": source}
                }
            )
            relevant = retriever.invoke(question)
            if relevant:
                context_docs = expand_retrieved_chunks(
                    self.vectorstore,
                    relevant,
                    neighbors=1,
                )
                context = "\n\n".join(
                    merge_chunk_overlaps(group)
                    for _, group in group_chunks_by_source_page(context_docs)
                )
                prompt = RAG_PROMPT.format(context=context, question=question)
                try:
                    results[source] = self._invoke_llm(prompt)
                except Exception as error:
                    results[source] = (
                        "La réponse de l’assistant n’a pas pu être générée pour ce document. "
                        f"Type d’erreur : {type(error).__name__}. Réessayez plus tard."
                    )
            else:
                results[source] = "Aucune information pertinente trouvée dans ce document."

        return results

    def reset(self):
        """Vide le vectorstore et repart de zéro."""
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.vectorstore = None
        self.ingested_docs = []

    @property
    def n_documents(self) -> int:
        return len(self.ingested_docs)

    @property
    def total_chunks(self) -> int:
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return sum(d['n_chunks'] for d in self.ingested_docs)
