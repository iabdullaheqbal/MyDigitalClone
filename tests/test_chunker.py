import pytest

from clone.ingestion.chunker import chunk_text


def test_chunker_splits_long_paragraphs_with_word_overlap() -> None:
    chunks = chunk_text("one two three four five six", chunk_words=4, overlap_words=2)

    assert chunks == ["one two three four", "three four five six"]


def test_chunker_packs_paragraphs_and_carries_a_small_overlap() -> None:
    chunks = chunk_text(
        "one two\n\nthree four\n\nfive six seven",
        chunk_words=4,
        overlap_words=1,
    )

    assert chunks == ["one two\n\nthree four", "four\n\nfive six seven"]


def test_chunker_keeps_faq_question_and_answer_together_when_they_fit() -> None:
    faq = "Q: What is this?\nA: A short answer."

    assert chunk_text(faq, chunk_words=20, overlap_words=4) == [faq]


def test_chunker_is_empty_safe_and_validates_limits() -> None:
    assert chunk_text(" \n\n ") == []
    with pytest.raises(ValueError, match="chunk_words"):
        chunk_text("words", chunk_words=0)
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_text("words", chunk_words=2, overlap_words=2)
