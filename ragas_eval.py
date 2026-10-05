"""
ragas_eval.py — Automated RAG Pipeline Evaluation using RAGAS-style metrics.

Implements reference-free and reference-based evaluation metrics for RAG:
  - Faithfulness   : is the answer grounded in the retrieved context?
  - Answer Relevancy: is the answer relevant to the question?
  - Context Precision: are the retrieved chunks relevant to the question?
  - Context Recall  : are all needed facts present in the retrieved chunks?

This module provides a lightweight, dependency-free implementation that
mirrors the RAGAS framework API without requiring the ragas package itself.
For production use, replace with: from ragas import evaluate

References:
    Es et al. (2023) "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    https://arxiv.org/abs/2309.15217
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGSample:
    """
    A single RAG evaluation sample.

    Parameters
    ----------
    question        : user query sent to the RAG pipeline.
    answer          : generated answer from the LLM.
    contexts        : list of retrieved document chunks used for generation.
    ground_truth    : reference answer (required for context recall only).
    """
    question:     str
    answer:       str
    contexts:     List[str]
    ground_truth: Optional[str] = None


@dataclass
class RAGEvalResult:
    """Aggregated evaluation results for a set of RAG samples."""
    faithfulness:       float = 0.0
    answer_relevancy:   float = 0.0
    context_precision:  float = 0.0
    context_recall:     float = 0.0
    n_samples:          int   = 0
    details:            List[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"RAGEvalResult("
            f"faithfulness={self.faithfulness:.3f}, "
            f"answer_relevancy={self.answer_relevancy:.3f}, "
            f"context_precision={self.context_precision:.3f}, "
            f"context_recall={self.context_recall:.3f}, "
            f"n={self.n_samples})"
        )

    def as_dict(self) -> dict:
        return {
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall":    round(self.context_recall, 4),
            "n_samples":         self.n_samples,
        }


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase word-level tokenisation, punctuation removed."""
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def faithfulness_score(answer: str, contexts: List[str]) -> float:
    """
    Measure how well the answer is grounded in the retrieved contexts.

    Approximation: fraction of answer tokens that appear in at least one
    retrieved context chunk (token overlap proxy).

    Returns a score in [0, 1].
    """
    if not answer.strip() or not contexts:
        return 0.0
    context_tokens = set()
    for ctx in contexts:
        context_tokens |= _tokenize(ctx)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def answer_relevancy_score(question: str, answer: str) -> float:
    """
    Measure how relevant the answer is to the question.

    Approximation: Jaccard similarity between question and answer token sets.
    (In production, replace with embedding cosine similarity.)

    Returns a score in [0, 1].
    """
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    intersection = q_tokens & a_tokens
    union = q_tokens | a_tokens
    return len(intersection) / len(union)


def context_precision_score(question: str, contexts: List[str]) -> float:
    """
    Measure what proportion of retrieved contexts are relevant to the question.

    Approximation: fraction of context chunks with non-zero overlap with
    the question token set.

    Returns a score in [0, 1].
    """
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    relevant = sum(1 for ctx in contexts if q_tokens & _tokenize(ctx))
    return relevant / len(contexts)


def context_recall_score(ground_truth: str, contexts: List[str]) -> float:
    """
    Measure whether the retrieved context contains the facts in the ground truth.

    Approximation: fraction of ground-truth tokens covered by the context pool.

    Returns a score in [0, 1]. Returns 0.0 if ground_truth is None.
    """
    if not ground_truth or not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    context_tokens = set()
    for ctx in contexts:
        context_tokens |= _tokenize(ctx)
    covered = gt_tokens & context_tokens
    return len(covered) / len(gt_tokens)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_rag_pipeline(samples: List[RAGSample]) -> RAGEvalResult:
    """
    Evaluate a list of RAG samples and return aggregated metrics.

    Parameters
    ----------
    samples : list of RAGSample objects.

    Returns
    -------
    RAGEvalResult with mean scores across all samples.
    """
    if not samples:
        return RAGEvalResult(n_samples=0)

    details = []
    faithfulness_scores     = []
    answer_relevancy_scores = []
    context_precision_scores = []
    context_recall_scores   = []

    for sample in samples:
        faith = faithfulness_score(sample.answer, sample.contexts)
        rel   = answer_relevancy_score(sample.question, sample.answer)
        prec  = context_precision_score(sample.question, sample.contexts)
        rec   = context_recall_score(sample.ground_truth or "", sample.contexts)

        faithfulness_scores.append(faith)
        answer_relevancy_scores.append(rel)
        context_precision_scores.append(prec)
        context_recall_scores.append(rec)

        details.append({
            "question":         sample.question[:80],
            "faithfulness":     round(faith, 4),
            "answer_relevancy": round(rel,   4),
            "context_precision": round(prec, 4),
            "context_recall":   round(rec,   4),
        })

    n = len(samples)
    return RAGEvalResult(
        faithfulness=       sum(faithfulness_scores)      / n,
        answer_relevancy=   sum(answer_relevancy_scores)  / n,
        context_precision=  sum(context_precision_scores) / n,
        context_recall=     sum(context_recall_scores)    / n,
        n_samples=n,
        details=details,
    )
