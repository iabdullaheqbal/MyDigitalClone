from clone.generation.answerer import MAX_COMPLETION_TOKENS, Answerer
from clone.generation.llm_client import GroqRateLimitError, GroqRequestError
from clone.types import Chunk, RetrievalHit, Turn


class FakeLLM:
    def __init__(
        self,
        results: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = iter(results or ["I built a small model."])
        self.error = error
        self.messages: list[list[dict[str, str]]] = []
        self.request_options: list[tuple[int, float]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.messages.append(messages)
        self.request_options.append((max_tokens, temperature))
        if self.error is not None:
            raise self.error
        return next(self.results)


def _hit() -> RetrievalHit:
    chunk = Chunk("faq#0", "faq.md", "I built a small model.", 0)
    return RetrievalHit(chunk, 1.0, 1.0, 0.8, 1, 1)


def test_answerer_passes_context_evidence_topics_and_history_to_the_llm() -> None:
    llm = FakeLLM()
    answerer = Answerer(llm, ("machine-learning pipelines", "backend engineering"))

    result = answerer.answer(
        "standalone",
        "follow-up",
        [_hit()],
        [Turn("user", "Earlier question")],
        "strong",
    )

    assert result.answer == "I built a small model."
    assert result.used_llm and result.error is None
    assert llm.request_options == [(MAX_COMPLETION_TOKENS, 0.2)]
    assert MAX_COMPLETION_TOKENS >= 1024
    assert "first person" in llm.messages[0][0]["content"]
    assert (
        "TOPICS:\nmachine-learning pipelines, backend engineering" in llm.messages[0][-1]["content"]
    )
    assert "EVIDENCE:\nstrong" in llm.messages[0][-1]["content"]
    assert "Earlier question" in llm.messages[0][-2]["content"]


def test_answerer_retries_once_with_stricter_grounding_then_returns_template() -> None:
    llm = FakeLLM(["I use Python for projects.", "I use Python for projects."])
    result = Answerer(llm).answer("query", "What do I use?", [_hit()], [], "weak")

    assert len(llm.messages) == 2
    assert "STRICT CHECK" in llm.messages[1][0]["content"]
    assert result.guardrail_flags == ("ungrounded_capitalized_term",)
    assert "I don't have that in my notes" not in result.answer
    assert any(word in result.answer for word in ("right now", "available", "verify"))
    assert not result.used_llm


def test_answerer_uses_safe_templates_without_key_or_after_api_errors() -> None:
    no_key = Answerer(None).answer("query", "question", [_hit()], [], "strong")
    assert not no_key.used_llm
    assert no_key.error == "Answer generation is not configured"
    assert any(word in no_key.answer for word in ("right now", "available", "verify"))
    assert any(
        topic in no_key.answer
        for topic in (
            "retrieval, RAG, and reranking",
            "AI products and project work",
            "local LLMs and automation",
        )
    )

    failed = Answerer(FakeLLM(error=GroqRequestError("server unavailable"))).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert failed.error == "The answer service is temporarily unavailable"
    assert "server unavailable" not in failed.answer

    limited = Answerer(FakeLLM(error=GroqRateLimitError("rate limit"))).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert limited.error == "The answer service is temporarily rate-limited"
