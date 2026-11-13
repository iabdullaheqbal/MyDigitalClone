from clone.generation.answerer import MAX_COMPLETION_TOKENS, Answerer
from clone.generation.llm_client import (
    GroqEmptyOutputError,
    GroqRateLimitError,
    GroqRequestError,
    GroqTimeoutError,
)
from clone.types import AnswerReason, Chunk, RetrievalHit, Turn


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
    assert MAX_COMPLETION_TOKENS == 700
    assert "first person" in llm.messages[0][0]["content"]
    assert (
        "TOPICS:\nmachine-learning pipelines or backend engineering"
        in llm.messages[0][-1]["content"]
    )
    assert "EVIDENCE:\nstrong" in llm.messages[0][-1]["content"]
    assert "Earlier question" in llm.messages[0][-2]["content"]


def test_answerer_repairs_unsupported_sentence_without_regeneration() -> None:
    llm = FakeLLM(["I use Python for projects.", "I use Python for projects."])
    result = Answerer(llm).answer("query", "What do I use?", [_hit()], [], "weak")

    assert len(llm.messages) == 1
    assert result.guardrail_flags
    assert result.reason is AnswerReason.GROUNDEDNESS_FAILED
    assert "I don't have that in my notes" not in result.answer
    assert not result.used_llm


def test_answerer_uses_safe_templates_without_key_or_after_api_errors() -> None:
    no_key = Answerer(None).answer("query", "question", [_hit()], [], "strong")
    assert not no_key.used_llm
    assert no_key.error == "Answer generation is not configured"
    assert no_key.reason is AnswerReason.NO_KEY
    assert "From faq.md:" in no_key.answer

    failed = Answerer(FakeLLM(error=GroqRequestError("server unavailable"))).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert failed.error == "The answer service is temporarily unavailable"
    assert "server unavailable" not in failed.answer
    assert "From faq.md:" in failed.answer
    assert failed.reason is AnswerReason.LLM_HTTP_ERROR

    limited = Answerer(FakeLLM(error=GroqRateLimitError(25))).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert limited.error == "The answer service is temporarily rate-limited"
    assert limited.reason is AnswerReason.RATE_LIMITED
    assert limited.retry_after_seconds == 25
    assert "try again in about 25 seconds" in limited.answer
    assert "From faq.md:" in limited.answer

    empty = Answerer(FakeLLM(error=GroqEmptyOutputError())).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert empty.reason is AnswerReason.EMPTY_LLM_OUTPUT
    assert "From faq.md:" in empty.answer

    timed_out = Answerer(FakeLLM(error=GroqTimeoutError())).answer(
        "query", "question", [_hit()], [], "strong"
    )
    assert timed_out.reason is AnswerReason.TIMEOUT
    assert "From faq.md:" in timed_out.answer


def test_aggregate_claim_retries_unsupported_count_then_uses_hedged_examples() -> None:
    chunk = Chunk("codean#0", "codean.md", "CODEAN is my final year project.", 0)
    hit = RetrievalHit(chunk, 1.0, 1.0, 0.8, 1, 1)
    llm = FakeLLM(
        [
            "I have completed seven projects, including CODEAN.",
            "I have completed 7 projects, including CODEAN.",
        ]
    )

    result = Answerer(llm).answer(
        "How many projects have you done yet?",
        "How many projects have you done yet?",
        [hit],
        [],
        "strong",
    )

    assert len(llm.messages) == 1
    assert "do not give a total" in llm.messages[0][0]["content"]
    assert not result.used_llm
    assert "CODEAN" in result.answer
    assert "Among the projects" in result.answer or "project examples" in result.answer
    assert "seven" not in result.answer.casefold()
    assert not any(character.isdigit() for character in result.answer)
