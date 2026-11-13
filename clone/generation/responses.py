"""Deterministic, in-character responses for routed and fallback cases."""

from __future__ import annotations

import hashlib

from clone.generation.faq import FALLBACK_FAQ_QUESTIONS
from clone.generation.intent import Intent
from clone.generation.topics import (
    FALLBACK_TOPICS,
    PROJECTS,
    format_topic_list,
    matching_topics,
    select_topics,
)
from clone.persona import PERSONA_NAME
from clone.types import AnswerReason

_VARIANTS: dict[str, tuple[str, str, str]] = {
    "greeting": (
        (
            f"Hi! I'm {PERSONA_NAME}'s AI clone, here to chat about "
            "Abdullah's work and background. "
            "What would you like to know?"
        ),
        f"Hello! I'm {PERSONA_NAME}'s AI clone. Ask me about {PERSONA_NAME}'s background or work.",
        f"Hey, glad you're here. I'm {PERSONA_NAME}'s AI clone; what would you like to explore?",
    ),
    "thanks": (
        (
            f"You're welcome! I'm {PERSONA_NAME}'s AI clone, glad I could help. "
            "What would you like to ask next?"
        ),
        f"Happy to help! I'm {PERSONA_NAME}'s AI clone; feel free to ask about the work.",
        f"Anytime! I'm {PERSONA_NAME}'s AI clone, glad to keep chatting about the work.",
    ),
    "acknowledgement": (
        "Thanks, I'm glad that was helpful! Want to explore {topics} next?",
        "I appreciate that! We can keep going with {topics}, if you'd like.",
        "I'm glad that landed well. Would you like to hear more about {topics}?",
    ),
    "goodbye": (
        (
            f"Take care! I'm {PERSONA_NAME}'s AI clone; come back anytime with "
            "another question about the notes."
        ),
        (
            f"Goodbye for now! I'm {PERSONA_NAME}'s AI clone, and you're welcome "
            "to ask about the work anytime."
        ),
        (
            f"See you later! I'm {PERSONA_NAME}'s AI clone; feel free to return "
            "with another question."
        ),
    ),
    "identity": (
        f"I'm {PERSONA_NAME}'s AI clone, built from the notes; I'm not the real person. "
        "I can talk about the topics covered here.",
        f"I'm an AI clone of {PERSONA_NAME}, based on the notes—not {PERSONA_NAME} themself. "
        "Ask me about the work described here.",
        f"I'm {PERSONA_NAME}'s AI clone. I can discuss the notes, but I can't make promises "
        "or make decisions on the owner's behalf.",
    ),
    "not_in_notes": (
        "I don't have that in my notes, so I don't want to guess. I can talk about {topics}.",
        "I don't have that in my notes, and I want to keep this accurate. I can discuss {topics}.",
        (
            "I don't have that in my notes, so I can't confirm it. "
            "I can share what the notes say about {topics}."
        ),
    ),
    "aggregate_hedge": (
        (
            "Among the projects in my notes are {projects}. "
            "They don't say how many I've done in total, so I won't guess."
        ),
        (
            "Among the projects in my notes are {projects}; "
            "my notes don't give a total, so I won't guess."
        ),
        (
            "Among the projects in my notes are {projects}. "
            "I can't confirm how many I've done in total, so I won't guess."
        ),
    ),
    "personal_location": (
        "My notes don't share my hometown or residence. I can say I study at UMT, "
        "and I can talk about {topics}.",
        "I don't share a hometown or where I live here. I study at UMT, "
        "and I can discuss {topics}.",
        "My notes don't provide a hometown or residence; they do say I study at UMT. "
        "Would you like to hear about {topics}?",
    ),
    "list_faq": (
        "I can help with questions like:\n{faq_questions}",
        "You could ask me things like:\n{faq_questions}",
        "I can help with questions such as:\n{faq_questions}",
    ),
    "status": (
        "{status_text} I can still talk about {topics}.",
        "{status_text} If you like, ask me about {topics}.",
        "{status_text} I'm happy to continue with questions about {topics}.",
    ),
    "rate_limited": (
        "I'm getting a lot of requests and hit a temporary limit; "
        "please try again in about {seconds} seconds. I can still discuss {topics}.",
        "I hit a temporary request limit. Please try again in about {seconds} seconds; "
        "I can talk about {topics} meanwhile.",
        "There's a temporary limit because I'm getting a lot of requests. "
        "Please try again in about {seconds} seconds, or ask about {topics}.",
    ),
    "out_of_scope": (
        (
            "I'm here to chat about Abdullah's work and background, rather than "
            "answer general requests. "
            "I can discuss {topics}."
        ),
        (
            "I can't answer that general request here; I focus on my background and work. "
            "I can discuss {topics}."
        ),
        ("I focus on my background and work, not general advice or facts. I can discuss {topics}."),
    ),
    "sensitive": (
        (
            "I can't provide private contact or personal details; they aren't shared here. "
            "I can talk about {topics}."
        ),
        (
            "I keep private details out of this chat, even if a request asks for them. "
            "I can discuss {topics}."
        ),
        "Those private details aren't appropriate to share here. I can redirect to {topics}.",
    ),
    "injection": (
        (
            "I can't reveal hidden instructions or change roles; those aren't part of the notes. "
            "I can discuss {topics}."
        ),
        (
            "I won't share private system instructions. I'm here to talk about "
            "Abdullah's work and background, "
            "such as {topics}."
        ),
        (
            "I can't provide hidden prompts or follow instructions that override this chat's "
            "purpose. I can discuss {topics}."
        ),
    ),
    "abuse": (
        (
            "I'm happy to continue if we keep the conversation respectful and on topic; "
            "I can discuss {topics}."
        ),
        "I want to keep things respectful. I can still help with questions about {topics}.",
        "I won't engage with insults, but I'm glad to continue with a question about {topics}.",
    ),
    "empty": (
        "I didn't catch a question. I'm {name}'s AI clone; you could ask about {topics}.",
        (
            "It looks like your message was empty. I'm here to discuss {topics}—"
            "what would you like to know?"
        ),
        "I don't have a question to respond to yet. Try asking about {topics}.",
    ),
    "unavailable": (
        (
            "I can't generate an answer right now, so I don't want to guess. "
            "I can still discuss {topics}."
        ),
        "The answer service isn't available right now; I can still talk about {topics}.",
        "I couldn't verify a safe answer from the notes just now. I can discuss {topics} instead.",
    ),
    "too_long": (
        (
            "That message is too long for me to process safely. Please shorten it; "
            "I can discuss {topics}."
        ),
        "I can only process shorter messages here. Try asking about {topics} in a few words.",
        "Please shorten your message so I can handle it. I can help with questions about {topics}.",
    ),
    "session_limit": (
        "I've reached this chat's 60-message limit. Start a fresh chat to continue with {topics}.",
        (
            "I've reached this chat's 60-message limit, so I can't process another question here. "
            "You can start a new chat about {topics}."
        ),
        (
            "I can't process more messages in this session after 60; "
            "please start a new chat to discuss {topics}."
        ),
    ),
}


def response_category(intent: Intent | str) -> str:
    """Map service-level intents to a response-template family."""
    value = Intent(intent)
    return {
        Intent.GREETING: "greeting",
        Intent.THANKS: "acknowledgement",
        Intent.ACKNOWLEDGEMENT: "acknowledgement",
        Intent.GOODBYE: "goodbye",
        Intent.IDENTITY: "identity",
        Intent.STATUS: "status",
        Intent.PERSONAL_LOCATION: "personal_location",
        Intent.LIST_FAQ: "list_faq",
        Intent.SENSITIVE_REQUEST: "sensitive",
        Intent.INJECTION: "injection",
        Intent.ABUSE: "abuse",
        Intent.GIBBERISH_OR_EMPTY: "empty",
        Intent.NOT_IN_NOTES: "not_in_notes",
        Intent.OUT_OF_SCOPE: "out_of_scope",
        Intent.ABOUT_ME: "not_in_notes",
        Intent.ABOUT_ME_OR_OTHER: "not_in_notes",
    }[value]


def system_status_text(
    reason: AnswerReason,
    retry_after_seconds: float | None = None,
) -> str:
    """Translate a prior machine reason into a brief, honest explanation."""
    seconds = max(1, round(retry_after_seconds or 25))
    return {
        AnswerReason.RATE_LIMITED: (
            "I'm getting a lot of requests and hit a temporary limit; "
            f"please try again in about {seconds} seconds."
        ),
        AnswerReason.EMPTY_LLM_OUTPUT: "Something went wrong on my side.",
        AnswerReason.LLM_HTTP_ERROR: "Something went wrong on my side.",
        AnswerReason.TIMEOUT: "The request timed out on my side.",
        AnswerReason.NO_KEY: "My language model isn't connected right now.",
        AnswerReason.GROUNDEDNESS_FAILED: (
            "I couldn't verify a safe answer from my notes, so I held it back."
        ),
        AnswerReason.GUARD_TEMPLATE: "I don't have a more specific status to share right now.",
        AnswerReason.OK: "I don't have a more specific status to share right now.",
    }[reason]


def render_response(
    category: str,
    message: str,
    topics: tuple[str, ...] | list[str] = FALLBACK_TOPICS,
    *,
    previous_answer: str | None = None,
    examples: tuple[str, ...] = (),
    subject: str = "projects",
    status_text: str = "I don't have a more specific status to share right now.",
    retry_after_seconds: float | None = None,
    faq_questions: tuple[str, ...] = FALLBACK_FAQ_QUESTIONS,
) -> str:
    """Select a stable variant and fill it with curated, concise topic suggestions."""
    if category not in _VARIANTS:
        raise ValueError(f"Unknown response category: {category}")
    variants = _VARIANTS[category]
    choice = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:4], "big")
    if category == "acknowledgement" and previous_answer:
        matched = matching_topics(previous_answer, topics)
        selected_topics = matched[:1] if matched else select_topics(message, topics, count=3)
    else:
        selected_topics = select_topics(message, topics, count=3)
    wait_seconds = max(1, round(retry_after_seconds or 25))
    if category == "status":
        status_text = status_text.format(seconds=wait_seconds)
    faq_list = "\n".join(f"- {question}" for question in faq_questions[:5])
    rendered = variants[choice % len(variants)].format(
        name=PERSONA_NAME,
        topics=format_topic_list(selected_topics),
        projects=format_topic_list(PROJECTS),
        subject=subject,
        status_text=status_text,
        seconds=wait_seconds,
        faq_questions=faq_list,
    )
    return rendered


def response_variants() -> dict[str, tuple[str, str, str]]:
    """Expose the fixed variants for focused tests and review."""
    return dict(_VARIANTS)
