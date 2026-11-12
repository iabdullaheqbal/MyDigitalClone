from clone.retrieval.tokenizer import tokenize


def test_tokenizer_lowercases_extracts_unicode_words_and_removes_stopwords() -> None:
    assert tokenize("Why is CODEAN useful? naïve ML!") == ["codean", "useful", "naïve", "ml"]


def test_tokenizer_handles_punctuation_and_empty_text() -> None:
    assert tokenize("BM25/RRF, v1.2") == ["bm25", "rrf", "v1", "2"]
    assert tokenize("the and of") == []
