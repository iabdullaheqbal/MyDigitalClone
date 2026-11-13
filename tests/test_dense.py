import sys
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from clone.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from clone.types import Chunk


class FixedEncoder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> NDArray[np.float32]:
        vector = self.vector.copy()
        if normalize_embeddings:
            vector /= np.linalg.norm(vector)
        return np.tile(vector, (len(sentences), 1))


def _chunks(count: int) -> list[Chunk]:
    return [
        Chunk(f"notes#{index}", "notes.md", f"chunk {index}", index)
        for index in range(count)
    ]


def test_dense_scores_equal_cosine_for_normalized_vectors() -> None:
    chunks = _chunks(3)
    embeddings = np.asarray([[1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    retriever = DenseRetriever(chunks, embeddings, FixedEncoder([1, 0]))

    np.testing.assert_allclose(retriever.scores("query"), [1.0, 0.0, 0.8])
    assert [index for index, _ in retriever.search("query", 2)] == [0, 2]


def test_dense_validates_embedding_and_query_shapes() -> None:
    chunks = _chunks(2)
    with pytest.raises(ValueError, match="incompatible"):
        DenseRetriever(chunks, np.ones((1, 2), dtype=np.float32), FixedEncoder([1, 0]))
    with pytest.raises(ValueError, match="normalized"):
        DenseRetriever(chunks, np.asarray([[2, 0], [0, 1]], dtype=np.float32), FixedEncoder([1, 0]))

    retriever = DenseRetriever(
        chunks,
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        FixedEncoder([1, 0, 0]),
    )
    with pytest.raises(ValueError, match="Query embedding shape"):
        retriever.search("query", 1)


def test_sentence_transformer_encoder_forces_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, device: str) -> None:
            received["model_name"] = model_name
            received["device"] = device

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    SentenceTransformerEncoder("test-model")._load_model()

    assert received == {"model_name": "test-model", "device": "cpu"}
