"""Orchestrate BM25, dense retrieval, RRF, and optional reranking."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from clone.retrieval.bm25 import BM25Retriever
from clone.retrieval.dense import DenseRetriever
from clone.retrieval.fusion import rrf
from clone.types import RetrievalHit, RetrievalMode


class Reranker(Protocol):
    """Injected optional cross-encoder interface."""

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]: ...


def _rank_scores(scores: NDArray[np.float64] | NDArray[np.float32]) -> list[int]:
    return [
        int(index)
        for index in np.argsort(-scores, kind="stable")
    ]


class HybridRetriever:
    """Retrieve with one mode and preserve both component scores and ranks."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        if len(bm25.chunks) != len(dense.chunks) or any(
            sparse.id != vector.id
            for sparse, vector in zip(bm25.chunks, dense.chunks, strict=True)
        ):
            raise ValueError("BM25 and dense retrievers must index the same chunks in order")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.bm25 = bm25
        self.dense = dense
        self.chunks = bm25.chunks
        self.rrf_k = rrf_k
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]:
        """Return top-k evidence for a retrieval mode, optionally reranked."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        if use_reranker and self.reranker is None:
            raise ValueError("Reranking was requested but no reranker was configured")

        mode = RetrievalMode(mode)
        bm25_scores = self.bm25.scores(query)
        dense_scores = self.dense.scores(query)
        bm25_order = [
            index
            for index in _rank_scores(bm25_scores)
            if bm25_scores[index] > 0
        ]
        dense_order = _rank_scores(dense_scores)
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, start=1)}

        rrf_scores: dict[int, float] = {}
        if mode is RetrievalMode.BM25_ONLY:
            ranked_indices = bm25_order
        elif mode is RetrievalMode.DENSE_ONLY:
            ranked_indices = dense_order
        else:
            fused = rrf(
                [bm25_order[:candidates], dense_order[:candidates]],
                k=self.rrf_k,
            )
            ranked_indices = [index for index, _ in fused]
            rrf_scores = dict(fused)

        candidate_limit = candidates if use_reranker else k
        hits = []
        for index in ranked_indices[:candidate_limit]:
            bm25_score = float(bm25_scores[index])
            dense_score = float(dense_scores[index])
            if mode is RetrievalMode.BM25_ONLY:
                score = bm25_score
            elif mode is RetrievalMode.DENSE_ONLY:
                score = dense_score
            else:
                score = rrf_scores[index]
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[index],
                    score=score,
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                    bm25_rank=bm25_ranks.get(index),
                    dense_rank=dense_ranks.get(index),
                    rrf_score=rrf_scores.get(index),
                )
            )

        if use_reranker:
            assert self.reranker is not None
            hits = self.reranker.rerank(query, hits, top_n=k)
        return hits[:k]
