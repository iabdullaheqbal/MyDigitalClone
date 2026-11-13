"""Detection and guardrails for aggregate or superlative claims."""

from __future__ import annotations

import re

_AGGREGATE_QUESTION = re.compile(
    r"\b(?:how\s+many|how\s+much|total|all|every|list\s+all|best|most|ever|only|first|last)\b",
    re.IGNORECASE,
)
_AGGREGATE_SUBJECT = re.compile(r"\b(projects?|skills?|topics?)\b", re.IGNORECASE)


def is_aggregate_claim(message: str) -> bool:
    """Return whether a question asks for a total, exhaustive set, or superlative."""
    return bool(_AGGREGATE_QUESTION.search(message))


def aggregate_subject(message: str) -> str:
    """Choose a supported noun for a no-count hedge template."""
    match = _AGGREGATE_SUBJECT.search(message)
    if not match:
        return "items"
    subject = match.group().casefold()
    return {
        "project": "projects",
        "projects": "projects",
        "skill": "skills",
        "skills": "skills",
        "topic": "topics",
        "topics": "topics",
    }[subject]
