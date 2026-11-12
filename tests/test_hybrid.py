import numpy as np
from numpy.typing import NDArray

from clone.retrieval.dense import DenseRetriever
from clone.retrieval.hybrid import HybridRetriever
from clone.types import Chunk, RetrievalHit, RetrievalMode


class FixedScoresBM25:
    def __init__(self, chunks: list[Chunk], scores: list[float]) -> None:
        self.chunks = chunks
        self._scores = np.asarray(scores, dtype=np.float64)

    def scores(self, query: str) -> NDArray[np.float64]:
        assert query == "question"
        return self._scores


class FixedQueryEncoder:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> NDArray[np.float32]:
        assert normalize_embeddings
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        return np.tile(query, (len(sentences), 1))


class RecordingReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        assert query == "question"
        self.called = True
        return list(reversed(hits))[:top_n]


def _retriever(reranker: RecordingReranker | None = None) -> HybridRetriever:
    chunks = [
        Chunk("a", "notes.md", "A", 0),
        Chunk("b", "notes.md", "B", 1),
        Chunk("c", "notes.md", "C", 2),
        Chunk("d", "notes.md", "D", 3),
    ]
    embeddings = np.asarray(
        [[0.8, 0.6], [1.0, 0.0], [0.9, np.sqrt(0.19)], [0.5, np.sqrt(0.75)]],
        dtype=np.float32,
    )
    bm25 = FixedScoresBM25(chunks, [4.0, 3.0, 2.0, 0.0])
    dense = DenseRetriever(chunks, embeddings, FixedQueryEncoder())
    return HybridRetriever(bm25, dense, reranker=reranker)


def test_modes_have_component_metadata_and_fusion_changes_the_order() -> None:
    retriever = _retriever()
    bm25 = retriever.retrieve("question", RetrievalMode.BM25_ONLY, k=3)
    dense = retriever.retrieve("question", RetrievalMode.DENSE_ONLY, k=3)
    hybrid = retriever.retrieve("question", RetrievalMode.HYBRID_RRF, k=3, candidates=4)

    assert [hit.chunk.id for hit in bm25] == ["a", "b", "c"]
    assert [hit.chunk.id for hit in dense] == ["b", "c", "a"]
    assert [hit.chunk.id for hit in hybrid] == ["b", "a", "c"]
    assert all(hit.bm25_rank is not None and hit.dense_rank is not None for hit in hybrid)
    assert all(hit.rrf_score is not None for hit in hybrid)
    assert all(hit.rerank_score is None for hit in hybrid)


def test_reranker_is_opt_in_and_populates_its_score() -> None:
    reranker = RecordingReranker()
    retriever = _retriever(reranker)
    retriever.retrieve("question", RetrievalMode.HYBRID_RRF, k=2)
    assert not reranker.called

    results = retriever.retrieve(
        "question",
        RetrievalMode.HYBRID_RRF,
        k=2,
        candidates=4,
        use_reranker=True,
    )
    assert reranker.called
    assert [hit.chunk.id for hit in results] == ["d", "c"]
