from clone.generation.answerer import ABSTENTION, Answerer
from clone.generation.llm_client import GroqRequestError
from clone.types import Chunk, RetrievalHit, Turn


class FakeLLM:
    def __init__(self, result: str = "I built it.", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.messages: list[dict[str, str]] | None = None
        self.request_options: tuple[int, float] | None = None

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.messages = messages
        self.request_options = (max_tokens, temperature)
        if self.error is not None:
            raise self.error
        return self.result


def _hit(bm25_score: float = 1.0, dense_score: float = 0.1) -> RetrievalHit:
    chunk = Chunk("faq#0", "faq.md", "A grounded passage.", 0)
    return RetrievalHit(chunk, 1.0, bm25_score, dense_score, 1, 1)


def test_answerer_declines_without_calling_the_llm() -> None:
    llm = FakeLLM()

    answer, used_llm, declined, error = Answerer(llm, dense_min=0.2).answer(
        "unknown",
        "Tell me about an unknown fact",
        [_hit(bm25_score=0, dense_score=0.1)],
        [],
    )

    assert (answer, used_llm, declined, error) == (ABSTENTION, False, True, None)
    assert llm.messages is None


def test_answerer_passes_current_context_and_history_to_the_llm() -> None:
    llm = FakeLLM("A grounded answer.")

    result = Answerer(llm).answer(
        "standalone",
        "follow-up",
        [_hit()],
        [Turn("user", "Earlier question")],
    )

    assert result == ("A grounded answer.", True, False, None)
    assert llm.messages is not None
    assert llm.request_options == (512, 0.2)
    assert "first person" in llm.messages[0]["content"]
    assert "A grounded passage." in llm.messages[-1]["content"]
    assert "Earlier question" in llm.messages[-2]["content"]


def test_answerer_uses_safe_passage_fallback_without_key_or_on_api_error() -> None:
    answerer = Answerer(None)
    no_key = answerer.answer("query", "question", [_hit()], [])
    assert no_key[1:] == (False, False, "Groq API key is not configured")
    assert "A grounded passage." in no_key[0]

    failed = Answerer(FakeLLM(error=GroqRequestError("server unavailable"))).answer(
        "query",
        "question",
        [_hit()],
        [],
    )
    assert failed[1:] == (False, False, "server unavailable")
    assert "server unavailable" in failed[0]
