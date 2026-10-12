"""Regression tests for document ingestion and chat invocation."""
from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from core.rag_pipeline import RAGPipeline


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
