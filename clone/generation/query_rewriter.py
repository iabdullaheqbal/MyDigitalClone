"""Rewrite follow-up questions for retrieval."""

from __future__ import annotations

import logging
import re

from clone.generation.llm_client import ChatClient, GroqError
from clone.generation.prompts import rewrite_messages
from clone.types import Turn

logger = logging.getLogger(__name__)
_REFERENCE = re.compile(
    r"\b(?:it|that|this|they|those|there|why|how\s+come|the\s+project|"
    r"tell\s+me\s+more|and\s+\w+)\b",
    re.IGNORECASE,
)


def should_rewrite(question: str, history: list[Turn]) -> bool:
    """Rewrite only short or reference-dependent messages with history."""
    return bool(history) and (len(question.split()) <= 3 or bool(_REFERENCE.search(question)))


def normalize_query(question: str) -> str:
    """Normalize whitespace and case before cache lookup and retrieval."""
    return " ".join(question.casefold().split())


def rewrite(history: list[Turn], question: str, llm: ChatClient) -> str:
    """Return a standalone query, preserving the original on API failure."""
    if not history:
        return question
    try:
        result = llm.chat(
            rewrite_messages(history[-2:], question),
            max_tokens=256,
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
    return " ".join(result.split()) or question
