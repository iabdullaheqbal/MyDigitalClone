"""Privacy, injection, and input-shape checks without retaining message text."""

from __future__ import annotations

from dataclasses import dataclass

from clone.generation.intent import Intent, route_intent

MAX_INPUT_LENGTH = 2_000
MAX_SESSION_MESSAGES = 60


@dataclass(frozen=True, slots=True)
class InputCheck:
    """Sanitized classification data for one incoming message."""

    intent: Intent
    intent_override: Intent | None
    flags: tuple[str, ...]


def check_input(message: str) -> InputCheck:
    """Return intent and safe category flags; never include the message itself."""
    intent = route_intent(message)
    flags: list[str] = []
    if len(message) > MAX_INPUT_LENGTH:
        flags.append("too_long")
        intent = Intent.OUT_OF_SCOPE
    if intent is Intent.GIBBERISH_OR_EMPTY:
        flags.append("empty_or_gibberish")
    if intent is Intent.INJECTION:
        flags.append("injection_detected")
    if intent is Intent.SENSITIVE_REQUEST:
        flags.append("sensitive_request")
    if intent is Intent.ABUSE:
        flags.append("abusive_language")
    override = (
        intent
        if intent
        in {
            Intent.GIBBERISH_OR_EMPTY,
            Intent.INJECTION,
            Intent.SENSITIVE_REQUEST,
            Intent.ABUSE,
            Intent.OUT_OF_SCOPE,
        }
        else None
    )
    return InputCheck(intent, override, tuple(flags))
