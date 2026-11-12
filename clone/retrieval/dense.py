"""Cosine retrieval with normalized sentence-transformer embeddings."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from clone.types import Chunk


class Encoder(Protocol):
    """Interface implemented by production and test encoders."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> NDArray[np.float32]: ...


class SentenceTransformerEncoder:
    """Lazily load a CPU sentence-transformer model on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts and return a float32 matrix."""
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)
        result = self._load_model().encode(
            sentences,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(result, dtype=np.float32)


class DenseRetriever:
    """Rank chunks by cosine similarity using normalized vectors and numpy."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
        encoder: Encoder,
    ) -> None:
        if not chunks:
            raise ValueError("Dense retrieval requires at least one chunk")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
            raise ValueError(
                f"Embedding shape {matrix.shape} is incompatible with {len(chunks)} chunks"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Embeddings contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0) or not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Dense retrieval requires non-zero normalized embeddings")

        self.chunks = list(chunks)
        self.embeddings = matrix
        self.encoder = encoder

    def scores(self, query: str) -> NDArray[np.float32]:
        """Return cosine similarities in original corpus order."""
        query_embeddings = np.asarray(
            self.encoder.encode([query], normalize_embeddings=True),
            dtype=np.float32,
        )
        if (
            query_embeddings.ndim != 2
            or query_embeddings.shape != (1, self.embeddings.shape[1])
        ):
            raise ValueError(
                "Query embedding shape "
                f"{query_embeddings.shape} does not match index dimension "
                f"{self.embeddings.shape[1]}"
            )
        query_vector = query_embeddings[0]
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains non-finite values")
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            raise ValueError("Query embedding has zero length")
        return np.asarray(self.embeddings @ (query_vector / norm), dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-k cosine scores with stable ties by corpus position."""
        if k < 0:
            raise ValueError("k must not be negative")
        if k == 0:
            return []
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(index), float(scores[index])) for index in order]
