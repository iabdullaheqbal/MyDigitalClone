"""Grounded answer generation with safe no-key and API-error fallbacks."""

from __future__ import annotations

from clone.generation.guard import should_decline
from clone.generation.llm_client import ChatClient, GroqError
from clone.generation.prompts import answer_messages
from clone.types import RetrievalHit, Turn

ABSTENTION = "I don't have that in my notes."


class Answerer:
    """Generate an answer from current-turn evidence only."""

    def __init__(self, llm: ChatClient | None, dense_min: float = 0.20) -> None:
        self.llm = llm
        self.dense_min = dense_min

    def answer(
        self,
        standalone_query: str,
        question: str,
        hits: list[RetrievalHit],
        history: list[Turn],
    ) -> tuple[str, bool, bool, str | None]:
        """Return answer text, LLM usage, decline status, and safe error text."""
        if should_decline(hits, self.dense_min):
            return ABSTENTION, False, True, None
        if self.llm is None:
            return self._fallback(hits, "Groq API key is not configured"), False, False, (
                "Groq API key is not configured"
            )

        messages = answer_messages(standalone_query, question, hits, history[-6:])
        try:
            answer = self.llm.chat(messages, max_tokens=512, temperature=0.2).strip()
        except GroqError as error:
            return self._fallback(hits, str(error)), False, False, str(error)
        if not answer:
            error = "Groq returned an empty answer"
            return self._fallback(hits, error), False, False, error
        return answer, True, False, None

    @staticmethod
    def _fallback(hits: list[RetrievalHit], reason: str) -> str:
        if not hits:
            return f"I can't generate an answer right now: {reason}."
        closest = hits[0].chunk
        return (
            f"I can't generate an answer right now: {reason}. "
            f"The closest note I found is from {closest.source}:\n\n{closest.text}"
        )
