from clone.retrieval.bm25 import BM25Retriever
from clone.types import Chunk


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(f"notes#{index}", "notes.md", text, index) for index, text in enumerate(texts)]


def test_bm25_ranks_rare_keyword_and_returns_scores() -> None:
    retriever = BM25Retriever(
        _chunks(
            "The pseudonym neural retrieval term appears",
            "General retrieval notes",
            "Other machine learning notes",
        )
    )

    results = retriever.search("pseudonym", k=2)

    assert results[0][0] == 0
    assert results[0][1] > 0
    assert len(retriever.scores("pseudonym")) == 3


def test_bm25_filters_zero_scores_and_uses_stable_ties() -> None:
    retriever = BM25Retriever(
        _chunks(
            "rare term",
            "rare term",
            "different words",
            "more unrelated text",
            "another unrelated passage",
        )
    )

    assert [index for index, _ in retriever.search("rare", k=3)] == [0, 1]
    assert retriever.search("absent", k=3) == []
    assert retriever.search("rare", k=0) == []
