"""Pure Reciprocal Rank Fusion implementation."""

from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse 1-based rankings with RRF and stable first-seen tie-breaking."""
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for ranking in rankings:
        seen_in_ranking = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen_in_ranking:
                continue
            seen_in_ranking.add(item)
            if item not in first_seen:
                first_seen[item] = len(first_seen)
                scores[item] = 0.0
            scores[item] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda result: (-result[1], first_seen[result[0]]))
