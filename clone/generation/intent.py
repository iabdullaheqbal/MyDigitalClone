"""Rule-based routing for messages that should bypass retrieval or generation."""

from __future__ import annotations

import re
from enum import StrEnum
from unicodedata import category as unicode_category


class Intent(StrEnum):
    """Supported behavior-layer message categories."""

    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    IDENTITY = "identity"
    STATUS = "status"
    PERSONAL_LOCATION = "personal_location"
    ACKNOWLEDGEMENT = "acknowledgement"
    SENSITIVE_REQUEST = "sensitive_request"
    INJECTION = "injection"
    ABUSE = "abuse"
    GIBBERISH_OR_EMPTY = "gibberish_or_empty"
    ABOUT_ME_OR_OTHER = "about_me_or_other"
    ABOUT_ME = "about_me"
    NOT_IN_NOTES = "not_in_notes"
    OUT_OF_SCOPE = "out_of_scope"


_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b.{0,50}\b(?:instructions?|rules?|prompt)\b"
    r"|\b(?:reveal|print|show|repeat|leak)\b.{0,40}\b(?:system|hidden|developer)\b"
    r"|\bdeveloper\s+mode\b|\bact\s+as\s+(?:someone|somebody|another|a different)\b"
    r"|\bpretend\s+(?:to\s+be|you'?re)\b",
    re.IGNORECASE,
)
_STATUS = re.compile(
    r"\b(?:why\s+(?:is|isn't|isnt|was|did)\s+(?:the\s+)?answer\s+service\b"
    r"|\bare\s+you\s+broken\b|\bwhy\s+did\s+you\s+fail\b|\bwhat\s+happened\b"
    r"|\bwhy\s+(?:is|isn't|isnt)\s+(?:the\s+)?(?:answer|service)\b"
    r"|\b(?:answer\s+service|service)\s+(?:isn't|isnt|not)\s+available\b)",
    re.IGNORECASE,
)
_PERSONAL_LOCATION = re.compile(
    r"\bwhere\s+(?:are\s+you\s+from|do\s+you\s+live|are\s+you\s+from|"
    r"do\s+you\s+belong|are\s+you\s+based)\b"
    r"|\bfrom\s+where\s+(?:are\s+you|do\s+you\s+belong)\b"
    r"|\bwhere\s+do\s+you\s+come\s+from\b",
    re.IGNORECASE,
)
_SENSITIVE = re.compile(
    r"\b(?:phone|mobile|cell)\s*(?:number|no\.?|#)\b"
    r"|\b(?:home\s+)?address\b|\bhome\s+location\b"
    r"|\b(?:cnic|national\s+id|identity\s+number|passport\s+number|social\s+security)\b"
    r"|\b(?:password|passcode|api\s+key|secret\s+key|bank\s+(?:details?|info(?:rmation)?)|"
    r"account\s+number|"
    r"credit\s+card)\b"
    r"|\bprivate\s+(?:info(?:rmation)?|details?)\b.{0,80}"
    r"\b(?:family|friends?|classmates?)\b"
    r"|\b(?:family|friends?|classmates?)\b.{0,45}\b(?:private|personal|contact|address|"
    r"phone|email|number|secret|medical|financial|salary|health|where\s+(?:they|he|she)\s+live)\b",
    re.IGNORECASE,
)
_ABUSE = re.compile(
    r"\b(?:fuck(?:ing)?|shit|idiot|stupid|moron|shut\s+up|you(?:'re| are)\s+useless)\b",
    re.IGNORECASE,
)
_IDENTITY = re.compile(
    r"\bwho\s+are\s+you\b|\bwhat\s+are\s+you\b"
    r"|\bare\s+you\s+(?:a\s+)?(?:real(?:\s+person)?|human|(?:an?\s+)?ai)\b"
    r"|\bare\s+you\s+an?\s+artificial\s+intelligence\b",
    re.IGNORECASE,
)
_GREETINGS = re.compile(
    r"^(?:hi(?:\s+there)?|hello(?:\s+there)?|hey(?:\s+there)?|hiya|good\s+morning|"
    r"good\s+afternoon|good\s+evening|what'?s\s+up|how\s+are\s+you(?:\s+doing)?|"
    r"how'?s\s+it\s+going)[!.?\s]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"\b(?:thanks|thank\s+you|thx|much\s+appreciated)\b", re.IGNORECASE)
_ACKNOWLEDGEMENT = re.compile(
    r"^(?:(?:ahn+|ah+|oh|hm+|uhm+|umm+)\s+)?"
    r"(?:(?:that'?s|that\s+is)\s+)?"
    r"(?:great|nice|wow|cool|awesome|ok(?:ay)?|got\s+it|hm+|lol|haha+|"
    r"excellent|perfect|sweet|interesting|right)[.!?\s]*$",
    re.IGNORECASE,
)
_GOODBYE = re.compile(
    r"\b(?:bye|goodbye|see\s+you|take\s+care|later|farewell)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]+")
_VALID_LOW_VOWEL_WORDS = {"rhythm", "rhythms", "strengths"}


def _is_acknowledgement(message: str) -> bool:
    normalized = re.sub(r"([a-z])\1{2,}", r"\1", message.strip().casefold())
    normalized = re.sub(r"[^\w\s']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if _ACKNOWLEDGEMENT.fullmatch(normalized):
        return True
    if any(character.isalnum() for character in normalized):
        return False
    return any(unicode_category(character) in {"So", "Sk"} for character in message)


def _is_gibberish(message: str) -> bool:
    stripped = message.strip()
    if not stripped or not _WORD.search(stripped):
        return True
    words = _WORD.findall(stripped)
    if len(stripped) >= 4 and len(set(stripped.casefold())) == 1:
        return True
    if len(stripped) >= 8 and len(words) == 1:
        vowel_count = len(re.findall(r"[aeiou]", stripped, re.I))
        return vowel_count <= 1 and stripped.casefold() not in _VALID_LOW_VOWEL_WORDS
    return False


def route_intent(message: str) -> Intent:
    """Classify high-confidence intents; leave all other messages evidence-led."""
    normalized = message.strip()
    if _INJECTION.search(normalized):
        return Intent.INJECTION
    if _SENSITIVE.search(normalized):
        return Intent.SENSITIVE_REQUEST
    if _ABUSE.search(normalized):
        return Intent.ABUSE
    if _IDENTITY.search(normalized):
        return Intent.IDENTITY
    if _STATUS.search(normalized):
        return Intent.STATUS
    if _PERSONAL_LOCATION.search(normalized):
        return Intent.PERSONAL_LOCATION
    if _GREETINGS.fullmatch(normalized):
        return Intent.GREETING
    if _THANKS.search(normalized):
        return Intent.ACKNOWLEDGEMENT
    if _GOODBYE.search(normalized):
        return Intent.GOODBYE
    if _is_acknowledgement(normalized):
        return Intent.ACKNOWLEDGEMENT
    if _is_gibberish(normalized):
        return Intent.GIBBERISH_OR_EMPTY
    return Intent.ABOUT_ME_OR_OTHER


def is_about_me_question(message: str) -> bool:
    """Distinguish personal-note questions from general or unrelated requests."""
    return bool(
        re.search(
            r"\b(?:you|your|yours|yourself|me|my|mine|myself|abdullah|his|he)\b"
            r"|\b(?:what|which|where|when|who|how)\s+(?:do|did|does|is|are)\s+"
            r"(?:you|he|abdullah)\b",
            message,
            re.IGNORECASE,
        )
    )
