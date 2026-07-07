"""
RAG Pipeline — Document Intelligence
Stack : LangChain + Mistral AI + ChromaDB

Architecture :
  Document(s) → Chunking → Embeddings (Mistral) → ChromaDB
                                                        ↓
  Question → Embedding → Retrieval (top-k chunks) → LLM (Mistral) → Réponse
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ── Prompt système ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un assistant expert en analyse de documents techniques et scientifiques.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Cite les passages pertinents entre guillemets quand c'est utile.
Réponds dans la même langue que la question.

Contexte :
{context}

Question : {question}

Réponse :"""
)

SUMMARY_PROMPT = """Analyse ce document et produis un résumé structuré en JSON avec :
- "title": titre ou sujet principal du document
- "summary": résumé en 3-5 phrases
- "key_points": liste de 5 points clés
- "domain": domaine (ex: Machine Learning, Finance, Médecine...)
- "language": langue du document
- "complexity": niveau de complexité (Débutant / Intermédiaire / Expert)

Document (extrait) :
{text}

Réponds UNIQUEMENT avec le JSON, sans texte autour."""


class RAGPipeline:
    """
    Pipeline RAG complet :
    - Ingestion de documents (PDF, DOCX, TXT)
    - Chunking intelligent avec RecursiveTextSplitter
    - Embeddings via Mistral AI
    - Stockage vectoriel ChromaDB (persistant)
    - Retrieval + génération avec Mistral
    """

    def __init__(
        self,
        api_key: str,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "mistral-small-latest",
        embedding_model: str = "mistral-embed",
        top_k: int = 4,
        temperature: float = 0.1,
    ):
        os.environ["MISTRAL_API_KEY"] = api_key
        self.api_key = api_key
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.top_k = top_k

        # Embeddings Mistral
        self.embeddings = MistralAIEmbeddings(
            model=embedding_model,
            mistral_api_key=api_key
        )

        # LLM Mistral
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            mistral_api_key=api_key,
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
        """Hash unique pour éviter les doublons."""
        return hashlib.md5(f"{filename}{len(content)}".encode()).hexdigest()[:8]

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

    def ingest(self, file_path: str, file_bytes: bytes = None) -> Dict[str, Any]:
        """
        Ingère un document dans le vectorstore.
        Retourne des stats sur l'ingestion.
        """
        filename = Path(file_path).name
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
            })

        # Ajout au vectorstore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(chunks)

        stats = {
            'filename': filename,
            'n_pages': len(docs),
            'n_chunks': n_chunks,
            'avg_chunk_size': int(sum(len(c.page_content) for c in chunks) / max(n_chunks, 1)),
            'total_chars': sum(len(c.page_content) for c in chunks),
        }
        self.ingested_docs.append(stats)
        return stats

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Pose une question sur les documents ingérés.
        Retourne la réponse + les sources + les chunks utilisés.
        """
        if self.vectorstore is None:
            raise ValueError("Aucun document ingéré. Charger des documents d'abord.")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance → diversité des résultats
            search_kwargs={"k": self.top_k, "fetch_k": self.top_k * 2}
        )

        # Récupérer les chunks pertinents
        relevant_docs = retriever.invoke(question)

        # Construire le contexte
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"chunk {doc.metadata.get('chunk_id', '?')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        # Générer la réponse
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = response.content

        result = {
            'question': question,
            'answer': answer,
            'n_sources': len(relevant_docs),
        }

        if return_sources:
            result['sources'] = [
                {
                    'source': doc.metadata.get('source', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 0),
                    'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                    'relevance_score': None,
                }
                for doc in relevant_docs
            ]

        return result

    def summarize(self, file_path: str = None, text: str = None) -> Dict[str, Any]:
        """
        Génère un résumé structuré d'un document.
        Peut prendre un fichier ou un texte brut.
        """
        if file_path:
            docs = self.load_document(file_path)
            full_text = "\n".join([d.page_content for d in docs])
        elif text:
            full_text = text
        else:
            raise ValueError("Fournir file_path ou text")

        # Limiter à ~4000 chars pour l'API
        excerpt = full_text[:4000]

        prompt = SUMMARY_PROMPT.format(text=excerpt)
        response = self.llm.invoke(prompt)

        import json
        try:
            # Extraire le JSON de la réponse
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            summary_data = json.loads(content)
        except Exception:
            summary_data = {
                "title": "Document analysé",
                "summary": response.content[:500],
                "key_points": [],
                "domain": "Non détecté",
                "language": "Français",
                "complexity": "Intermédiaire"
            }

        summary_data['full_text_length'] = len(full_text)
        summary_data['n_words'] = len(full_text.split())
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
                context = "\n\n".join([d.page_content for d in relevant])
                prompt = RAG_PROMPT.format(context=context, question=question)
                resp = self.llm.invoke(prompt)
                results[source] = resp.content
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
