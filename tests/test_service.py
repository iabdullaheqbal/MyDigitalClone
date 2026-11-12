import pytest

from clone.chat.session import ChatSession
from clone.service import CloneService
from clone.types import Chunk, RetrievalHit, RetrievalMode


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[list[dict[str, str]], int, float]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append((messages, max_tokens, temperature))
        return next(self.responses)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, RetrievalMode, int, int, bool]] = []
        chunk = Chunk("faq#0", "faq.md", "A grounded passage.", 0)
        self.hits = [RetrievalHit(chunk, 1.0, 1.0, 0.8, 1, 1)]

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]:
        self.calls.append((query, mode, k, candidates, use_reranker))
        return self.hits[:k]


def test_service_rewrites_follow_up_retrieves_and_records_exchange() -> None:
    llm = FakeLLM(["What did Veritas use?", "It used a task-specific model."])
    retriever = FakeRetriever()
    session = ChatSession()
    session.record_exchange("What is Veritas?", "A project.")
    service = CloneService(
        retriever,
        llm,
        candidates=12,
        history_turns=4,
    )

    result = service.answer(
        "What did it use?",
        session,
        mode=RetrievalMode.BM25_ONLY,
        top_k=3,
        use_reranker=True,
    )

    assert result.answer == "It used a task-specific model."
    assert result.standalone_query == "What did Veritas use?"
    assert result.mode is RetrievalMode.BM25_ONLY
    assert result.used_llm and not result.declined
    assert retriever.calls == [
        ("What did Veritas use?", RetrievalMode.BM25_ONLY, 3, 12, True)
    ]
    assert len(llm.calls) == 2
    assert session.history()[-2].content == "What did it use?"
    assert session.history()[-1].content == "It used a task-specific model."
    assert set(result.latency_ms) == {"query_rewrite", "retrieval", "generation", "total"}
    assert all(value >= 0 for value in result.latency_ms.values())


def test_service_without_groq_uses_grounded_fallback_and_original_query() -> None:
    retriever = FakeRetriever()
    session = ChatSession()
    service = CloneService(retriever, None)

    result = service.answer("  question  ", session)

    assert result.standalone_query == "question"
    assert result.answer.startswith("I can't generate an answer right now:")
    assert result.error == "Groq API key is not configured"
    assert not result.used_llm
    assert retriever.calls[0][0] == "question"
    assert session.history()[-2].content == "question"


def test_service_rejects_invalid_input_without_retrieving_or_mutating_session() -> None:
    retriever = FakeRetriever()
    session = ChatSession()
    service = CloneService(retriever, None)

    with pytest.raises(ValueError, match="Question must not be empty"):
        service.answer("  ", session)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        service.answer("question", session, top_k=0)

    assert retriever.calls == []
    assert session.history() == []


def test_service_validates_history_and_candidate_limits() -> None:
    retriever = FakeRetriever()

    with pytest.raises(ValueError, match="candidates must be at least 1"):
        CloneService(retriever, None, candidates=0)
    with pytest.raises(ValueError, match="history_turns must not be negative"):
        CloneService(retriever, None, history_turns=-1)
