"""
Unit tests for RAGAS-style RAG evaluation metrics.
"""
from __future__ import annotations

import pytest
from ragas_eval import (
    RAGSample,
    evaluate_rag_pipeline,
    faithfulness_score,
    answer_relevancy_score,
    context_precision_score,
    context_recall_score,
)

# ── Sample data ──────────────────────────────────────────────────────────────

GOOD_CONTEXT = [
    "GARCH models capture volatility clustering in financial time series.",
    "The GARCH(1,1) model has three parameters: omega, alpha, and beta.",
    "Covariance stationarity requires alpha + beta < 1 in GARCH models.",
]

QUESTION = "What are the parameters of the GARCH(1,1) model?"
ANSWER   = "The GARCH(1,1) model has three parameters: omega, alpha, and beta."
GROUND_TRUTH = "The parameters are omega, alpha, and beta."

IRRELEVANT_CONTEXT = [
    "The Eiffel Tower is located in Paris, France.",
    "Python was created by Guido van Rossum in 1991.",
]


# ── Faithfulness ─────────────────────────────────────────────────────────────

class TestFaithfulness:

    def test_perfect_faithfulness(self):
        """Answer fully derived from context => score near 1.0."""
        score = faithfulness_score(ANSWER, GOOD_CONTEXT)
        assert score > 0.5

    def test_irrelevant_context_lower_faithfulness(self):
        """Irrelevant context must yield lower faithfulness than relevant."""
        relevant   = faithfulness_score(ANSWER, GOOD_CONTEXT)
        irrelevant = faithfulness_score(ANSWER, IRRELEVANT_CONTEXT)
        assert relevant > irrelevant

    def test_empty_answer_returns_zero(self):
        assert faithfulness_score("", GOOD_CONTEXT) == 0.0

    def test_empty_context_returns_zero(self):
        assert faithfulness_score(ANSWER, []) == 0.0

    def test_score_in_unit_interval(self):
        score = faithfulness_score(ANSWER, GOOD_CONTEXT)
        assert 0.0 <= score <= 1.0


# ── Answer Relevancy ─────────────────────────────────────────────────────────

class TestAnswerRelevancy:

    def test_identical_strings_high_score(self):
        score = answer_relevancy_score("omega alpha beta", "omega alpha beta")
        assert score == 1.0

    def test_unrelated_question_low_score(self):
        score = answer_relevancy_score("What is the Eiffel Tower height?", ANSWER)
        assert score < 0.3

    def test_score_in_unit_interval(self):
        score = answer_relevancy_score(QUESTION, ANSWER)
        assert 0.0 <= score <= 1.0

    def test_empty_inputs_return_zero(self):
        assert answer_relevancy_score("", ANSWER) == 0.0
        assert answer_relevancy_score(QUESTION, "") == 0.0


# ── Context Precision ─────────────────────────────────────────────────────────

class TestContextPrecision:

    def test_fully_relevant_context_high_score(self):
        score = context_precision_score(QUESTION, GOOD_CONTEXT)
        assert score > 0.5

    def test_irrelevant_context_low_score(self):
        score = context_precision_score(QUESTION, IRRELEVANT_CONTEXT)
        assert score < 0.5

    def test_empty_context_returns_zero(self):
        assert context_precision_score(QUESTION, []) == 0.0

    def test_score_in_unit_interval(self):
        score = context_precision_score(QUESTION, GOOD_CONTEXT)
        assert 0.0 <= score <= 1.0


# ── Context Recall ────────────────────────────────────────────────────────────

class TestContextRecall:

    def test_ground_truth_covered_by_context(self):
        score = context_recall_score(GROUND_TRUTH, GOOD_CONTEXT)
        assert score > 0.5

    def test_irrelevant_context_lower_recall(self):
        relevant   = context_recall_score(GROUND_TRUTH, GOOD_CONTEXT)
        irrelevant = context_recall_score(GROUND_TRUTH, IRRELEVANT_CONTEXT)
        assert relevant > irrelevant

    def test_empty_ground_truth_returns_zero(self):
        assert context_recall_score("", GOOD_CONTEXT) == 0.0

    def test_none_ground_truth_returns_zero(self):
        assert context_recall_score(None, GOOD_CONTEXT) == 0.0


# ── Batch evaluation ─────────────────────────────────────────────────────────

class TestEvaluatePipeline:

    def test_returns_ragrevalresult(self):
        samples = [RAGSample(QUESTION, ANSWER, GOOD_CONTEXT, GROUND_TRUTH)]
        result = evaluate_rag_pipeline(samples)
        assert result.n_samples == 1

    def test_all_scores_in_unit_interval(self):
        samples = [
            RAGSample(QUESTION, ANSWER, GOOD_CONTEXT, GROUND_TRUTH),
            RAGSample("What is omega?", "Omega is the base volatility.", GOOD_CONTEXT[:1]),
        ]
        result = evaluate_rag_pipeline(samples)
        for score in [result.faithfulness, result.answer_relevancy,
                      result.context_precision, result.context_recall]:
            assert 0.0 <= score <= 1.0

    def test_empty_list_returns_zero_samples(self):
        result = evaluate_rag_pipeline([])
        assert result.n_samples == 0

    def test_as_dict_has_required_keys(self):
        samples = [RAGSample(QUESTION, ANSWER, GOOD_CONTEXT, GROUND_TRUTH)]
        d = evaluate_rag_pipeline(samples).as_dict()
        required = {"faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "n_samples"}
        assert required <= d.keys()

    def test_good_rag_outscores_bad_rag(self):
        """A pipeline with relevant context should beat one with irrelevant context."""
        good = evaluate_rag_pipeline([RAGSample(QUESTION, ANSWER, GOOD_CONTEXT, GROUND_TRUTH)])
        bad  = evaluate_rag_pipeline([RAGSample(QUESTION, ANSWER, IRRELEVANT_CONTEXT, GROUND_TRUTH)])
        assert good.faithfulness >= bad.faithfulness
        assert good.context_recall >= bad.context_recall
