"""Graded retrieval-evidence assessment."""

from __future__ import annotations

from typing import Literal

from clone.types import RetrievalHit

EvidenceLevel = Literal["strong", "weak", "none"]


def assess_evidence(
    hits: list[RetrievalHit],
    strong_cosine: float = 0.45,
    weak_cosine: float = 0.20,
) -> EvidenceLevel:
    """Grade current-turn evidence using cosine similarity and BM25 matches."""
    if not -1.0 <= weak_cosine <= strong_cosine <= 1.0:
        raise ValueError("Evidence thresholds must satisfy -1 <= weak <= strong <= 1")
    if not hits:
        return "none"

    best_dense = max(hit.dense_score for hit in hits)
    if best_dense >= strong_cosine:
        return "strong"
    if best_dense >= weak_cosine or any(hit.bm25_score > 0 for hit in hits):
        return "weak"
    return "none"
