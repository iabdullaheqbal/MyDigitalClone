"""Transient conversation state for one browser session."""

from __future__ import annotations

from dataclasses import dataclass, field

from clone.types import Turn


@dataclass(slots=True)
class ChatSession:
    """Keep one conversation's turns separate from all other sessions."""

    turns: list[Turn] = field(default_factory=list)
    messages_used: int = 0

    def history(self) -> list[Turn]:
        """Return a copy so callers cannot mutate the stored transcript."""
        return list(self.turns)

    @property
    def user_message_count(self) -> int:
        """Count accepted messages, including ones intentionally not retained."""
        return max(self.messages_used, sum(turn.role == "user" for turn in self.turns))

    def record_exchange(
        self,
        question: str,
        answer: str,
        *,
        retain_in_history: bool = True,
    ) -> None:
        """Count a completed exchange and optionally retain it for continuity."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        self.messages_used = self.user_message_count + 1
        if retain_in_history:
            self.turns.extend((Turn("user", question), Turn("assistant", answer)))

    def clear(self) -> None:
        """Remove all conversation history from this session."""
        self.turns.clear()
        self.messages_used = 0
