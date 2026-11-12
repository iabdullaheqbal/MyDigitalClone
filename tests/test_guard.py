from clone.generation.guard import should_decline
from clone.types import Chunk, RetrievalHit


def _hit(bm25_score: float, dense_score: float) -> RetrievalHit:
    chunk = Chunk("notes#0", "notes.md", "text", 0)
    return RetrievalHit(chunk, 0.0, bm25_score, dense_score, None, None)


def test_guard_declines_missing_or_irrelevant_evidence() -> None:
    assert should_decline([], dense_min=0.2)
    assert should_decline([_hit(0, 0.19)], dense_min=0.2)
    assert not should_decline([_hit(0, 0.2)], dense_min=0.2)
    assert not should_decline([_hit(0.01, -0.5)], dense_min=0.2)
