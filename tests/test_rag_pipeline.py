"""Regression tests for document ingestion and chat invocation."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from core.rag_pipeline import (
    RAGPipeline,
    expand_retrieved_chunks,
    merge_chunk_overlaps,
    select_summary_excerpt,
)


class FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.deleted_sources = []
        self.added_ids = []

    def delete(self, *, where):
        source = where["source"]
        self.deleted_sources.append(source)
        self.documents = {
            doc_id: doc for doc_id, doc in self.documents.items()
            if doc.metadata.get("source") != source
        }

    def add_documents(self, documents, *, ids):
        self.added_ids.append(ids)
        self.documents.update(zip(ids, documents))

    def get(self, *, where, include):
        documents = [
            (doc_id, document)
            for doc_id, document in self.documents.items()
            if document.metadata.get("source") == where["source"]
        ]
        return {
            "ids": [doc_id for doc_id, _ in documents],
            "documents": [document.page_content for _, document in documents],
            "metadatas": [document.metadata for _, document in documents],
        }

    def as_retriever(self, **kwargs):
        return SimpleNamespace(invoke=lambda question: [
            Document(page_content="Relevant context", metadata={"source": "report.txt"})
        ])


class FakeSplitter:
    def split_documents(self, documents):
        return [
            Document(page_content=text, metadata={})
            for text in documents[0].page_content.split("|")
        ]


def make_pipeline():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.vectorstore = FakeVectorStore()
    pipeline.embeddings = object()
    pipeline.splitter = FakeSplitter()
    pipeline.ingested_docs = []
    pipeline.persist_dir = "unused"
    pipeline.top_k = 4
    return pipeline


def test_reingesting_same_upload_does_not_increase_chunk_count(monkeypatch):
    pipeline = make_pipeline()
    monkeypatch.setattr(
        pipeline,
        "load_document",
        lambda file_path: [Document(page_content="first chunk|second chunk")],
    )

    first = pipeline.ingest("/tmp/random-a.txt", b"same bytes", source_name="report.txt")
    initial_ids = pipeline.vectorstore.added_ids[-1]
    initial_count = len(pipeline.vectorstore.documents)
    second = pipeline.ingest("/tmp/random-b.txt", b"same bytes", source_name="report.txt")

    assert first["filename"] == second["filename"] == "report.txt"
    assert len(initial_ids) == initial_count == 2
    assert pipeline.vectorstore.added_ids[-1] == initial_ids
    assert len(pipeline.vectorstore.documents) == initial_count
    assert pipeline.n_documents == 1
    assert pipeline.vectorstore.deleted_sources == ["report.txt", "report.txt"]


def test_reingesting_changed_content_replaces_old_chunks(monkeypatch):
    pipeline = make_pipeline()
    document_text = "old chunk|stale chunk"
    monkeypatch.setattr(
        pipeline,
        "load_document",
        lambda file_path: [Document(page_content=document_text)],
    )

    pipeline.ingest("/tmp/random.txt", b"old content", source_name="report.txt")
    old_ids = set(pipeline.vectorstore.documents)
    document_text = "replacement chunk"
    pipeline.ingest("/tmp/random.txt", b"new content", source_name="report.txt")

    assert len(pipeline.vectorstore.documents) == 1
    assert old_ids.isdisjoint(pipeline.vectorstore.documents)
    assert next(iter(pipeline.vectorstore.documents.values())).page_content == "replacement chunk"


def test_query_uses_configured_chat_model_instead_of_fallback():
    pipeline = make_pipeline()
    invocations = []

    class FakeLLM:
        def invoke(self, prompt):
            invocations.append(prompt)
            return SimpleNamespace(content="Generated answer")

    pipeline.llm = FakeLLM()
    result = pipeline.query("What does the report say?")

    assert result["answer"] == "Generated answer"
    assert len(invocations) == 1
    assert "What does the report say?" in invocations[0]


def test_query_expands_relevant_match_to_adjacent_chunks():
    pipeline = make_pipeline()
    store = pipeline.vectorstore
    for chunk_id, content in enumerate([
        "The study asked an important question",
        "and evaluated a new method",
        "on a clearly defined dataset",
        "with results that support the conclusion",
        "unrelated appendix text",
    ]):
        store.documents[f"doc:{chunk_id}"] = Document(
            page_content=content,
            metadata={"source": "report.pdf", "chunk_id": chunk_id, "page": 2},
        )
    store.as_retriever = lambda **kwargs: SimpleNamespace(invoke=lambda question: [
        store.documents["doc:2"]
    ])
    prompts = []

    class FakeLLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="Complete answer.")

    pipeline.llm = FakeLLM()
    result = pipeline.query("What was the result?")

    assert result["answer"] == "Complete answer."
    assert "evaluated a new method" in prompts[0]
    assert "clearly defined dataset" in prompts[0]
    assert "results that support the conclusion" in prompts[0]
    assert "unrelated appendix text" not in prompts[0]
    assert result["sources"][0]["content"] == "on a clearly defined dataset"
    assert result["sources"][0]["page"] == 2


def test_adjacent_chunks_merge_without_duplicate_overlap():
    chunks = [
        Document(page_content="The model produced a reliable result"),
        Document(page_content="reliable result across the full test set."),
    ]

    assert merge_chunk_overlaps(chunks) == (
        "The model produced a reliable result across the full test set."
    )


def test_query_marks_fallback_when_chat_model_fails():
    pipeline = make_pipeline()

    class FailingLLM:
        def invoke(self, prompt):
            raise RuntimeError("provider unavailable")

    pipeline.llm = FailingLLM()
    result = pipeline.query("What does the report say?")

    assert result["generation_failed"] is True
    assert result["answer"] is None
    assert result["generation_error_type"] == "RuntimeError"


def test_summary_samples_text_across_long_documents():
    text = "\n".join(
        f"SECTION_{index} " + ("x" * 2400)
        for index in range(20)
    )

    excerpt = select_summary_excerpt(text)

    assert len(excerpt) <= 16000
    assert "SECTION_0" in excerpt
    assert "SECTION_19" in excerpt
    assert any(f"SECTION_{index}" in excerpt for index in range(8, 13))


def test_summary_uses_all_indexed_text_and_returns_structured_fields(monkeypatch):
    pipeline = make_pipeline()
    text = "Opening section. " + ("Body content. " * 2000) + "Final conclusion."
    response = {
        "title": "Document title",
        "summary": "Executive summary.",
        "objectives": "Study objective.",
        "methodology": "Methods described.",
        "key_findings": "Main finding.",
        "conclusion": "Conclusion.",
        "limitations": "Non précisées.",
        "key_points": ["Point one", "Point two"],
        "domain": "Research",
        "language": "English",
        "complexity": "Intermédiaire",
    }
    prompts = []
    monkeypatch.setattr(
        pipeline,
        "_invoke_llm",
        lambda prompt: prompts.append(prompt) or json.dumps(response),
    )

    summary = pipeline.summarize(text=text, file_path="report.pdf")

    assert summary["title"] == "Document title"
    assert summary["n_words"] == len(text.split())
    assert summary["full_text_length"] == len(text)
    assert "Opening section." in prompts[0]
    assert "Final conclusion." in prompts[0]
    assert "[Extrait 8/8]" in prompts[0]


def test_summary_does_not_silently_return_raw_text_when_model_fails(monkeypatch):
    pipeline = make_pipeline()
    monkeypatch.setattr(
        pipeline,
        "_invoke_llm",
        lambda prompt: '{"summary": "incomplete"}',
    )

    with pytest.raises(ValueError, match="résumé généré est incomplet"):
        pipeline.summarize(text="A document with enough useful content.")
