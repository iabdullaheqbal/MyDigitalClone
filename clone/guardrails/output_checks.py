"""Output redaction and lexical checks for grounded, privacy-safe answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from clone.guardrails.privacy import redact_names

_PII_PATTERNS = (
    ("credential_redacted", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)),
    ("key_redacted", re.compile(r"\b(?:gsk|sk|ghp|hf|api)[_-][A-Za-z0-9_-]{16,}\b", re.I)),
    ("email_redacted", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("long_digit_run_redacted", re.compile(r"(?<!\d)\d{5,}(?!\d)")),
    (
        "phone_or_digits_redacted",
        re.compile(r"(?<!\w)(?:\+?\d[\d().\s-]{5,}\d)(?!\w)"),
    ),
    ("key_like_string_redacted", re.compile(r"\b[A-Za-z0-9_-]{32,}\b")),
)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NUMBER_WORD = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"dozen|couple)\b",
    re.IGNORECASE,
)
_QUANTIFIER = re.compile(
    r"\b(?:all|every|only|never|always|over|total|best|most|ever|first|last)\b"
    r"|\bmore\s+than\b",
    re.I,
)
_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z'-]{2,}\b")
_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|my|me)\b", re.IGNORECASE)
_COMMON_CAPITALIZED = {
    "A",
    "Abdullah",
    "AI",
    "An",
    "As",
    "But",
    "Can",
    "Could",
    "Do",
    "For",
    "Goodbye",
    "Hello",
    "Hey",
    "Hi",
    "I",
    "I'm",
    "If",
    "It",
    "It's",
    "Here",
    "Its",
    "My",
    "No",
    "Not",
    "Please",
    "So",
    "Thanks",
    "That",
    "The",
    "This",
    "There",
    "These",
    "Today",
    "We",
    "When",
    "What",
    "While",
    "You",
}
_PROMPT_LEAK_MARKERS = (
    "rules by situation",
    "owner to approve",
    "evidence_level:",
    "system prompt",
    "you are the ai clone of",
)


@dataclass(frozen=True, slots=True)
class OutputCheck:
    """Redacted output plus categorized validation results."""

    text: str
    flags: tuple[str, ...]
    acceptable: bool


def contains_pii(text: str) -> bool:
    """Return whether text contains a pattern that output redaction handles."""
    return any(pattern.search(text) for _, pattern in _PII_PATTERNS)


def redact_pii(text: str) -> str:
    """Redact contact details, long numbers, and key-like strings from display text."""
    for _, pattern in _PII_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def check_output(
    answer: str,
    notes: str,
    persona_facts: str = "AI clone",
    topics: Iterable[str] = (),
    third_party_names: Iterable[str] = (),
    require_first_person: bool = True,
) -> OutputCheck:
    """Redact PII and reject ungrounded or prompt-leaking model responses."""
    text = answer.strip()
    flags: list[str] = []
    if not text:
        flags.append("empty_answer")
    for flag, pattern in _PII_PATTERNS:
        text, replacements = pattern.subn("[redacted]", text)
        if replacements:
            flags.append(flag)
    text, names_redacted = redact_names(text, third_party_names)
    if names_redacted:
        flags.append("third_party_name_redacted")

    lowered = text.casefold()
    if "as an ai language model" in lowered or re.fullmatch(
        r"i don't have that in my notes[.! ]*", lowered
    ):
        flags.append("rejected_phrase")
    if any(marker in lowered for marker in _PROMPT_LEAK_MARKERS):
        flags.append("prompt_leak")
    if require_first_person and not _FIRST_PERSON.search(text):
        flags.append("missing_first_person")

    trusted_text = f"{notes}\n{persona_facts}\n{' '.join(topics)}"
    trusted = trusted_text.casefold()
    trusted_notes = notes.casefold()
    trusted_numbers = set(_NUMBER.findall(trusted))
    unsupported_numbers = [
        number for number in _NUMBER.findall(text) if number not in trusted_numbers
    ]
    if unsupported_numbers:
        flags.append("ungrounded_number")

    unsupported_number_words = [
        match.group().casefold()
        for match in _NUMBER_WORD.finditer(text)
        if not re.search(rf"\b{re.escape(match.group().casefold())}\b", trusted_notes)
    ]
    if unsupported_number_words:
        flags.append("ungrounded_number_word")

    unsupported_quantifiers = [
        match.group().casefold()
        for match in _QUANTIFIER.finditer(text)
        if not re.search(rf"\b{re.escape(match.group().casefold())}\b", trusted_notes)
    ]
    if unsupported_quantifiers:
        flags.append("ungrounded_quantifier")

    unsupported_terms = []
    for match in _CAPITALIZED.finditer(text):
        term = match.group()
        if term in _COMMON_CAPITALIZED:
            continue
        if not re.search(rf"\b{re.escape(term.casefold())}\b", trusted):
            unsupported_terms.append(term)
    if unsupported_terms:
        flags.append("ungrounded_capitalized_term")

    return OutputCheck(text, tuple(dict.fromkeys(flags)), not flags)


def repair_grounded_answer(
    answer: str,
    notes: str,
    *,
    persona_facts: str = "AI clone",
    topics: Iterable[str] = (),
    third_party_names: Iterable[str] = (),
) -> OutputCheck:
    """Keep only grounded sentences and return the resulting safe text."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    kept: list[str] = []
    flags: list[str] = []
    safe_redactions = {
        "credential_redacted",
        "key_redacted",
        "email_redacted",
        "long_digit_run_redacted",
        "phone_or_digits_redacted",
        "key_like_string_redacted",
        "third_party_name_redacted",
    }
    for sentence in sentences:
        if not sentence.strip():
            continue
        checked = check_output(
            sentence,
            notes,
            persona_facts=persona_facts,
            topics=topics,
            third_party_names=third_party_names,
            require_first_person=False,
        )
        flags.extend(checked.flags)
        fatal_flags = set(checked.flags) - safe_redactions - {"missing_first_person"}
        if not fatal_flags and checked.text.strip():
            kept.append(checked.text.strip())
    repaired = " ".join(kept)
    if not repaired or not _FIRST_PERSON.search(repaired):
        return OutputCheck("", tuple(dict.fromkeys(flags)), False)
    final = check_output(
        repaired,
        notes,
        persona_facts=persona_facts,
        topics=topics,
        third_party_names=third_party_names,
    )
    remaining = set(final.flags) - safe_redactions
    return OutputCheck(
        final.text,
        tuple(dict.fromkeys((*flags, *final.flags))),
        not remaining,
    )
