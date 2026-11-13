"""FAQ question extraction for indexed, no-LLM help responses."""

from __future__ import annotations

import re
from collections.abc import Iterable

from clone.ingestion.loader import Document

FALLBACK_FAQ_QUESTIONS = (
    "What are you studying and what do you focus on?",
    "What are you working on right now?",
    "What is CODEAN?",
    "What is Resona?",
    "What's your tech stack?",
)
_FAQ_QUESTION = re.compile(r"^\s*(?:\*\*)?Q:\s*(.+?)\s*(?:\*\*)?\s*$", re.MULTILINE)


def extract_faq_questions(documents: Iterable[Document], limit: int = 5) -> tuple[str, ...]:
    """Extract up to five visitor-facing question headings from faq.md."""
    if limit < 1:
        raise ValueError("FAQ question limit must be at least 1")
    for document in documents:
        if document.name.casefold() == "faq.md":
            questions = tuple(
                question.strip()
                for question in _FAQ_QUESTION.findall(document.text)
                if question.strip()
            )
            if questions:
                return questions[:limit]
    return FALLBACK_FAQ_QUESTIONS[:limit]
