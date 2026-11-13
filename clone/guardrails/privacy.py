"""Privacy filtering for third-party names associated with personal roles."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from clone.persona import PERSONA_NAME
from clone.types import RetrievalHit

_NAME = (
    r"(?:(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+)?"
    r"[A-Z][a-z]+(?:[-'][A-Za-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Za-z]+)?){0,2}"
)
_ROLE_NAMES = re.compile(
    rf"(?i:\b(?:my\s+)?(?:supervisor|advisor|adviser|mentor|teammates?|"
    rf"team\s+members?|colleagues?|project\s+partners?)\b)"
    rf"(?:\s+(?:is|are|was|were|named|called|include|includes|included))?"
    rf"\s*(?:[:,=-]\s*|\s+)(?P<names>{_NAME}(?:\s*(?:,|and)\s*{_NAME}){{0,2}})"
)
_HONORIFIC = re.compile(r"^(?:(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+)", re.IGNORECASE)
_ORG_PRODUCT_PREFIX = re.compile(
    r"\b(?:company|organization|organisation|university|project|product|lab)\s+"
    r"(?:named|called)?\s*",
    re.IGNORECASE,
)
_KNOWN_ORG_OR_PRODUCT_NAMES = {"codean", "resona", "umt"}


def extract_third_party_names(text: str) -> tuple[str, ...]:
    """Find names explicitly attached to supervisor, teammate, or colleague roles."""
    candidates: list[str] = []
    for match in _ROLE_NAMES.finditer(text):
        names = re.split(r"\s*,\s*|\s+and\s+", match.group("names"))
        for candidate in names:
            name = _HONORIFIC.sub("", candidate).strip()
            if (
                not name
                or name.casefold() == PERSONA_NAME.casefold()
                or name.casefold() in _KNOWN_ORG_OR_PRODUCT_NAMES
                or name.isupper()
                or name in candidates
            ):
                continue
            if re.search(
                rf"{_ORG_PRODUCT_PREFIX}{re.escape(name)}\b",
                text,
                re.IGNORECASE,
            ):
                continue
            candidates.append(name)
    return tuple(candidates)


def redact_names(text: str, names: Iterable[str]) -> tuple[str, bool]:
    """Replace known third-party names while leaving unrelated entities untouched."""
    changed = False
    for name in sorted(set(names), key=len, reverse=True):
        text, replacements = re.subn(
            rf"(?<!\w){re.escape(name)}(?!\w)",
            "[redacted]",
            text,
            flags=re.IGNORECASE,
        )
        changed = changed or replacements > 0
    return text, changed


def redact_retrieval_hits(
    hits: list[RetrievalHit], *, show_names: bool
) -> tuple[list[RetrievalHit], tuple[str, ...]]:
    """Sanitize retrieved passages for generation and display when privacy is enabled."""
    if show_names:
        return hits, ()

    names = tuple(
        dict.fromkeys(name for hit in hits for name in extract_third_party_names(hit.chunk.text))
    )
    sanitized = []
    for hit in hits:
        text, _ = redact_names(hit.chunk.text, names)
        sanitized.append(replace(hit, chunk=replace(hit.chunk, text=text)))
    return sanitized, names
