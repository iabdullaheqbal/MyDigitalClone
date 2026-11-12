from clone.types import Chunk, RetrievalHit, RetrievalMode, Turn


def test_shared_types_preserve_retrieval_metadata() -> None:
    chunk = Chunk(id="faq#0", source="faq.md", text="answer", index=0)
    hit = RetrievalHit(chunk, 0.5, 1.0, 0.2, 1, 2)

    assert hit.chunk.source == "faq.md"
    assert hit.bm25_rank == 1
    assert hit.dense_rank == 2
    assert hit.rerank_score is None
    assert RetrievalMode.HYBRID_RRF.value == "Hybrid (RRF)"
    assert Turn("user", "Hello").role == "user"
