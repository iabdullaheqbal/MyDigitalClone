"""Pre-generation relevance guard."""

from __future__ import annotations

from clone.types import RetrievalHit


def should_decline(hits: list[RetrievalHit], dense_min: float) -> bool:
    """Decline if there is no positive BM25 match and dense similarity is low."""
    if not hits:
        return True
    has_sparse_match = any(hit.bm25_score > 0 for hit in hits)
    best_dense = max(hit.dense_score for hit in hits)
    return not has_sparse_match and best_dense < dense_min
