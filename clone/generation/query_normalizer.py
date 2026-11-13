"""Normalize visitor phrasing for intent matching and retrieval."""

from __future__ import annotations

import re

_META_PREFIXES = (
    re.compile(r"^\s*what\s+do\s+(?:the\s+)?notes\s+say\s+about\s+", re.I),
    re.compile(r"^\s*according\s+to\s+(?:the\s+)?notes\b[:,]?\s*", re.I),
    re.compile(r"^\s*from\s+(?:your|the)\s+notes\b[:,]?\s*", re.I),
    re.compile(r"^\s*do\s+your\s+notes\s+mention\s+", re.I),
    re.compile(r"^\s*can\s+you\s+please\s+tell\s+me\b[:,]?\s*", re.I),
    re.compile(r"^\s*tell\s+me\s+(?=about\b)", re.I),
)
_META_FILLERS = re.compile(r"^(?:(?:a\s+)?bit\s+about|about)\s+", re.I)
_STUTTERED_LETTERS = re.compile(r"([a-z])\1{2,}", re.I)
_REPEATED_PUNCTUATION = re.compile(r"([!?.,;:])\1+")


def normalize_visitor_query(message: str) -> str:
    """Return a lowercase, de-metafied query without changing the displayed text."""
    normalized = message.strip().casefold()
    for pattern in _META_PREFIXES:
        normalized = pattern.sub("", normalized, count=1)
    normalized = _META_FILLERS.sub("", normalized, count=1)
    normalized = _STUTTERED_LETTERS.sub(r"\1", normalized)
    normalized = _REPEATED_PUNCTUATION.sub(r"\1", normalized)
    normalized = re.sub(r"\bmy\b", "your", normalized)
    normalized = re.sub(r"\bme\b", "you", normalized)
    return " ".join(normalized.split()).strip(" \t\r\n.,;:")
