"""
Unit tests for RAG-Document-Intelligence.

Tests hybrid search independently of any LLM API or vector store.
"""
from __future__ import annotations

import pytest
from hybrid_search import BM25Retriever, HybridRetriever, reciprocal_rank_fusion


CORPUS = [
    "The autoencoder learns to reconstruct normal sensor readings.",
    "Bearing friction causes vibration and temperature anomalies.",
    "PyTorch provides automatic differentiation for deep learning.",
    "BM25 is a classical sparse retrieval algorithm based on TF-IDF.",
    "Reciprocal Rank Fusion combines multiple ranked lists effectively.",
    "RAG systems retrieve relevant chunks before generating a response.",
    "Industrial predictive maintenance reduces unplanned downtime.",
    "Langchain provides abstractions for building LLM-powered applications.",
]


@pytest.fixture
def bm25(corpus: list[str] = CORPUS) -> BM25Retriever:
    return BM25Retriever().fit(corpus)


@pytest.fixture
def hybrid(corpus: list[str] = CORPUS) -> HybridRetriever:
    return HybridRetriever(corpus)


# -------------------------------------------------------------------------
# BM25 tests
# -------------------------------------------------------------------------

class TestBM25Retriever:

    def test_returns_k_results(self, bm25):
        results = bm25.retrieve("anomaly detection", k=3)
        assert len(results) == 3

    def test_result_indices_in_valid_range(self, bm25):
        results = bm25.retrieve("deep learning autoencoder", k=4)
        for idx, score in results:
            assert 0 <= idx < len(CORPUS), f"Index {idx} out of range"

    def test_scores_are_non_negative(self, bm25):
        results = bm25.retrieve("vibration temperature sensor", k=5)
        for _, score in results:
            assert score >= 0.0

    def test_sorted_descending(self, bm25):
        results = bm25.retrieve("retrieval ranked lists", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results must be ranked highest first"

    def test_relevant_doc_ranked_higher(self, bm25):
        """The BM25 top result for 'BM25 sparse retrieval' should match doc index 3."""
        results = bm25.retrieve("BM25 sparse retrieval", k=3)
        top_idx = results[0][0]
        assert top_idx == 3, f"Expected doc 3 at rank 1, got {top_idx}"

    def test_empty_query_does_not_crash(self, bm25):
        results = bm25.retrieve("", k=3)
        assert isinstance(results, list)


# -------------------------------------------------------------------------
# RRF tests
# -------------------------------------------------------------------------

class TestReciprocalRankFusion:

    def test_output_sorted_descending(self):
        fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 3]], k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_shared_doc_boosted(self):
        """A document appearing in both lists must have a higher score."""
        fused = dict(reciprocal_rank_fusion([[0, 1], [1, 2]], k=60))
        assert fused[1] > fused[0] or fused[1] > fused[2], (
            "Doc 1 (in both lists) must outrank docs that appear in only one list"
        )

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([], k=60) == []

    def test_single_list_passthrough(self):
        result = reciprocal_rank_fusion([[2, 0, 1]], k=60)
        indices = [i for i, _ in result]
        assert indices == [2, 0, 1]


# -------------------------------------------------------------------------
# HybridRetriever tests
# -------------------------------------------------------------------------

class TestHybridRetriever:

    def test_returns_k_results(self, hybrid):
        results = hybrid.retrieve("anomaly detection bearing", k=3)
        assert len(results) == 3

    def test_result_has_required_keys(self, hybrid):
        results = hybrid.retrieve("deep learning RAG", k=2)
        for r in results:
            assert "index"     in r
            assert "text"      in r
            assert "rrf_score" in r

    def test_rrf_scores_positive(self, hybrid):
        results = hybrid.retrieve("retrieval augmented generation", k=4)
        for r in results:
            assert r["rrf_score"] > 0.0

    def test_results_sorted_by_rrf_score(self, hybrid):
        results = hybrid.retrieve("predictive maintenance IoT", k=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_text_matches_corpus(self, hybrid):
        results = hybrid.retrieve("Langchain LLM applications", k=2)
        for r in results:
            assert r["text"] in CORPUS
