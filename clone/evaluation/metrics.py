"""Source-level retrieval metrics for the evaluation runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Precision, recall, and reciprocal rank at a fixed cutoff."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float


def score_retrieval(
    retrieved_sources: Sequence[str],
    relevant_sources: Sequence[str],
    k: int,
) -> RetrievalMetrics:
    """Score ranked source filenames against source-level relevance judgments."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_sources)
    if not relevant or any(not source for source in relevant):
        raise ValueError("At least one non-empty relevant source is required")
    ranked_sources = list(dict.fromkeys(retrieved_sources[:k]))
    matched = [source for source in ranked_sources if source in relevant]
    reciprocal_rank = (
        1.0 / (ranked_sources.index(matched[0]) + 1)
        if matched
        else 0.0
    )
    return RetrievalMetrics(
        precision_at_k=len(matched) / k,
        recall_at_k=len(matched) / len(relevant),
        reciprocal_rank_at_k=reciprocal_rank,
    )
