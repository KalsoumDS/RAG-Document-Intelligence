"""
hybrid_search.py — Hybrid semantic + BM25 retrieval for RAG-Document-Intelligence.

Combines dense vector similarity (FAISS via langchain-community) with
sparse BM25 lexical matching (rank_bm25). Scores are fused using
Reciprocal Rank Fusion (RRF) so neither modality dominates.

Usage:
    retriever = HybridRetriever(documents)
    results   = retriever.retrieve("your query", k=5)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# BM25 retriever (pure-Python, no external vector store required)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Okapi BM25 sparse retriever.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b  : float
        Document length normalisation factor (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b  = b
        self._corpus: list[str] = []
        self._tf: list[dict[str, float]] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0
        self._n: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def fit(self, documents: list[str]) -> "BM25Retriever":
        """Index the corpus."""
        self._corpus = documents
        self._n      = len(documents)
        tokenized    = [self._tokenize(d) for d in documents]
        self._avg_len = sum(len(t) for t in tokenized) / max(self._n, 1)

        # Term frequency per document
        self._tf = []
        for tok in tokenized:
            tf: dict[str, float] = {}
            for t in tok:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)

        # Document frequency
        self._df = {}
        for tf in self._tf:
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        # IDF (Robertson / Sparck Jones variant)
        self._idf = {
            term: math.log((self._n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self._df.items()
        }
        return self

    def score(self, query: str) -> np.ndarray:
        """Return BM25 score vector (length == corpus size)."""
        tokens = self._tokenize(query)
        scores = np.zeros(self._n, dtype=np.float32)
        for i, (tf, doc) in enumerate(zip(self._tf, self._corpus)):
            doc_len = sum(tf.values())
            for token in tokens:
                if token not in tf:
                    continue
                idf = self._idf.get(token, 0.0)
                f   = tf[token]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted descending."""
        scores  = self.score(query)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked document lists via RRF.

    Parameters
    ----------
    ranked_lists : list of ranked doc-index lists (most relevant first).
    k            : RRF constant (default 60, per Cormack et al. 2009).

    Returns
    -------
    List of (doc_index, rrf_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Dense + sparse hybrid retriever using Reciprocal Rank Fusion.

    In production this wraps a langchain VectorStore for dense retrieval.
    A lightweight FAISS-less path is provided for offline testing.

    Parameters
    ----------
    documents        : plain-text document chunks to index.
    vector_retriever : optional langchain BaseRetriever for dense search.
                       If None, falls back to cosine similarity on TF-IDF vectors.
    bm25_weight      : reserved for future weighted fusion (not yet used in RRF).
    """

    def __init__(
        self,
        documents: list[str],
        vector_retriever: Any | None = None,
        bm25_weight: float = 0.5,
    ) -> None:
        self.documents        = documents
        self.vector_retriever = vector_retriever
        self.bm25             = BM25Retriever().fit(documents)
        self._dense_matrix: np.ndarray | None = None

        if vector_retriever is None:
            self._build_tfidf_fallback()

    # ------------------------------------------------------------------
    def _build_tfidf_fallback(self) -> None:
        """Build a simple TF-IDF matrix for dense-like retrieval without FAISS."""
        vocab: dict[str, int] = {}
        tokenized = [d.lower().split() for d in self.documents]
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        n, v = len(self.documents), len(vocab)
        mat = np.zeros((n, v), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for t in tokens:
                mat[i, vocab[t]] += 1

        # L2 normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        self._dense_matrix = mat / norms
        self._vocab         = vocab

    def _dense_retrieve(self, query: str, k: int) -> list[int]:
        """Return top-k doc indices via cosine similarity (TF-IDF fallback)."""
        if self._dense_matrix is None or self._vocab is None:
            return []
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        for token in query.lower().split():
            if token in self._vocab:
                q_vec[self._vocab[token]] += 1
        norm = np.linalg.norm(q_vec) + 1e-8
        q_vec /= norm
        scores  = self._dense_matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top_idx]

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Hybrid retrieval: merge BM25 + dense rankings via RRF.

        Returns
        -------
        List of dicts with keys 'index', 'text', 'rrf_score'.
        """
        # BM25 ranking
        bm25_ranked = [idx for idx, _ in self.bm25.retrieve(query, k=k * 2)]

        # Dense ranking
        if self.vector_retriever is not None:
            dense_docs  = self.vector_retriever.get_relevant_documents(query)
            dense_ranked = [self.documents.index(d.page_content) for d in dense_docs
                            if d.page_content in self.documents]
        else:
            dense_ranked = self._dense_retrieve(query, k=k * 2)

        # Fuse
        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)

        return [
            {"index": idx, "text": self.documents[idx], "rrf_score": float(score)}
            for idx, score in fused[:k]
            if idx < len(self.documents)
        ]
