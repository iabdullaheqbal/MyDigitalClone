"""Transient conversation state for one browser session."""

from __future__ import annotations

from dataclasses import dataclass, field

from clone.types import Turn


@dataclass(slots=True)
class ChatSession:
    """Keep one conversation's turns separate from all other sessions."""

    turns: list[Turn] = field(default_factory=list)

    def history(self) -> list[Turn]:
        """Return a copy so callers cannot mutate the stored transcript."""
        return list(self.turns)

    def record_exchange(self, question: str, answer: str) -> None:
        """Append a completed user/assistant exchange."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        self.turns.extend((Turn("user", question), Turn("assistant", answer)))

    def clear(self) -> None:
        """Remove all conversation history from this session."""
        self.turns.clear()
