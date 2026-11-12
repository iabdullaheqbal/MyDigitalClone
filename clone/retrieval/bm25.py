"""BM25 sparse retrieval over the project tokenizer."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from clone.retrieval.tokenizer import tokenize
from clone.types import Chunk


class BM25Retriever:
    """Search chunks using Okapi BM25 and discard non-positive matches."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self.chunks = list(chunks)
        tokenized_corpus = [tokenize(chunk.text) for chunk in self.chunks]
        if not any(tokenized_corpus):
            raise ValueError("BM25 requires at least one searchable token")
        self._index = BM25Okapi(tokenized_corpus)

    def scores(self, query: str) -> NDArray[np.float64]:
        """Return one BM25 score per chunk in original corpus order."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return np.zeros(len(self.chunks), dtype=np.float64)
        return np.asarray(self._index.get_scores(query_tokens), dtype=np.float64)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return positive-score hits ordered by score, then corpus position."""
        if k < 0:
            raise ValueError("k must not be negative")
        if k == 0:
            return []
        scores = self.scores(query)
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores) if score > 0),
            key=lambda result: (-result[1], result[0]),
        )
        return ranked[:k]
