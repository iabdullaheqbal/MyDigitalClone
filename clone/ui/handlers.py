"""Gradio event handlers that delegate business logic to the service."""

from __future__ import annotations

from clone.chat.session import ChatSession
from clone.service import CloneService
from clone.types import RetrievalMode
from clone.ui.formatting import format_evidence


def handle_message(
    message: str,
    displayed_history: list[dict[str, str]] | None,
    session: ChatSession,
    mode: str,
    top_k: int,
    use_reranker: bool,
    *,
    service: CloneService,
) -> tuple[str, list[dict[str, str]], ChatSession, dict[str, object]]:
    """Process a user message and update only the caller's chat state."""
    if not isinstance(session, ChatSession):
        raise TypeError("The chat session state is invalid")
    question = message.strip()
    result = service.answer(
        question,
        session,
        mode=RetrievalMode(mode),
        top_k=int(top_k),
        use_reranker=use_reranker,
    )
    history = list(displayed_history or [])
    history.extend(
        (
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        )
    )
    return "", history, session, format_evidence(result)


def clear_conversation() -> tuple[list[dict[str, str]], ChatSession, dict[str, object]]:
    """Return fresh UI and session state when the user clears the chat."""
    return [], ChatSession(), {}
