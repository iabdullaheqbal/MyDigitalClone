from clone.generation.llm_client import GroqRequestError
from clone.generation.query_rewriter import rewrite
from clone.types import Turn


class FakeLLM:
    def __init__(self, result: str = "", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list[dict[str, str]], int, float]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append((messages, max_tokens, temperature))
        if self.error is not None:
            raise self.error
        return self.result


def test_rewriter_skips_empty_history_and_cleans_model_prefixes() -> None:
    llm = FakeLLM('Standalone question: "What did Veritas use?"')

    assert rewrite([], "What did it use?", llm) == "What did it use?"
    assert llm.calls == []
    assert rewrite([Turn("user", "I built Veritas.")], "What did it use?", llm) == (
        "What did Veritas use?"
    )
    assert llm.calls[0][1:] == (256, 0)


def test_rewriter_uses_only_recent_history_and_falls_back_on_api_failure() -> None:
    history = [Turn("user", f"turn-{index}") for index in range(8)]
    llm = FakeLLM("Standalone question: What happened?")

    assert rewrite(history, "Why?", llm) == "What happened?"
    transcript = llm.calls[0][0][-1]["content"]
    assert "turn-0" not in transcript
    assert "turn-6" in transcript
    assert "turn-7" in transcript

    broken = FakeLLM(error=GroqRequestError("offline"))
    assert rewrite(history, "Why?", broken) == "Why?"
