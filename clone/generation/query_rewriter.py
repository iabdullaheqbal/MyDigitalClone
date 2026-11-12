"""Rewrite follow-up questions for retrieval."""

from __future__ import annotations

import logging

from clone.generation.llm_client import ChatClient, GroqError
from clone.generation.prompts import rewrite_messages
from clone.types import Turn

logger = logging.getLogger(__name__)


def rewrite(history: list[Turn], question: str, llm: ChatClient) -> str:
    """Return a standalone query, preserving the original on API failure."""
    if not history:
        return question
    try:
        result = llm.chat(
            rewrite_messages(history[-6:], question),
            max_tokens=80,
            temperature=0,
        ).strip()
    except GroqError as error:
        logger.warning("Query rewrite failed: %s", type(error).__name__)
        return question

    result = result.strip().strip("\"' ")
    for prefix in ("Standalone question:", "Question:"):
        if result.casefold().startswith(prefix.casefold()):
            result = result[len(prefix) :].strip()
            break
    result = result.strip().strip("\"' ")
    return result or question
