"""Optional cross-encoder reranking of retrieved passages."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from clone.types import RetrievalHit


class CrossEncoder(Protocol):
    """Minimal prediction interface for injected cross-encoder models."""

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]: ...


class CrossEncoderReranker:
    """Load a cross-encoder only if reranking is explicitly requested."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: CrossEncoder | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder as SentenceCrossEncoder

            self._model = SentenceCrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Sort candidate hits by cross-encoder score and return top_n."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not hits:
            return []

        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = np.asarray(
            self._load_model().predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if scores.shape != (len(hits),) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned invalid reranking scores")

        reranked = [
            replace(hit, score=float(score), rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: -hit.score)[:top_n]
