from clone.generation.guard import assess_evidence
from clone.types import Chunk, RetrievalHit


def _hit(bm25_score: float, dense_score: float) -> RetrievalHit:
    chunk = Chunk("notes#0", "notes.md", "text", 0)
    return RetrievalHit(chunk, 0.0, bm25_score, dense_score, None, None)


def test_evidence_grading_combines_cosine_thresholds_and_bm25_matches() -> None:
    assert assess_evidence([]) == "none"
    assert assess_evidence([_hit(0, 0.19)]) == "none"
    assert assess_evidence([_hit(0, 0.20)]) == "weak"
    assert assess_evidence([_hit(0.01, -0.5)]) == "weak"
    assert assess_evidence([_hit(0, 0.45)]) == "strong"


def test_evidence_thresholds_must_be_ordered() -> None:
    try:
        assess_evidence([_hit(0, 0.2)], strong_cosine=0.1, weak_cosine=0.2)
    except ValueError as error:
        assert "thresholds" in str(error)
    else:
        raise AssertionError("Expected invalid evidence thresholds to fail")
