"""Curated topic labels derived from indexed document filenames."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from clone.ingestion.loader import Document

FALLBACK_TOPICS = (
    "my background and skills",
    "Resona",
    "CODEAN, my final year project",
    "my other projects",
    "how I think about retrieval, security and backend engineering",
    "my setup and how I work",
    "common questions",
)
PROJECTS = (
    "CODEAN",
    "Resona",
    "Veritas",
    "FocusOS",
    "CCDA",
    "an Urdu fake-news detector",
    "a CAPTCHA breaker",
)

_FILE_TOPICS = {
    "bio": "my background and skills",
    "cv_skills": "my background and skills",
    "resona": "Resona",
    "codean": "CODEAN, my final year project",
    "other_projects": "my other projects",
    "how_i_work": "my setup and how I work",
    "faq": "common questions",
    "opinions": "my opinions",
}
_NOTES_TOPIC = "how I think about retrieval, security and backend engineering"
_TOPIC_STOP_WORDS = {"a", "about", "and", "how", "i", "in", "my", "of", "or", "the", "to"}


def topics_for_sources(sources: Iterable[str]) -> list[str]:
    """Map source filenames to curated labels without exposing raw filenames."""
    topics: list[str] = []
    for source in sources:
        stem = Path(source).stem.casefold()
        topic = _NOTES_TOPIC if stem.startswith("notes_") else _FILE_TOPICS.get(stem)
        if topic and topic not in topics:
            topics.append(topic)
    return topics or list(FALLBACK_TOPICS)


def derive_topics(documents: Iterable[Document]) -> list[str]:
    """Build persisted topic labels from document filenames."""
    return topics_for_sources(document.name for document in documents)


def matching_topics(text: str, topics: Iterable[str] = FALLBACK_TOPICS) -> list[str]:
    """Return labels whose words occur in text, ordered by relevance."""
    available = tuple(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))
    if not available:
        return []
    query_words = set(re.findall(r"[a-z0-9]+", text.casefold())) - _TOPIC_STOP_WORDS
    scored = [
        (
            sum(
                word in query_words
                for word in re.findall(r"[a-z0-9]+", topic.casefold())
                if word not in _TOPIC_STOP_WORDS
            ),
            index,
            topic,
        )
        for index, topic in enumerate(available)
    ]
    offset = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rotated_index = {index: (index - offset) % len(available) for index in range(len(available))}
    return [
        topic
        for score, index, topic in sorted(
            scored, key=lambda item: (-item[0], rotated_index[item[1]])
        )
        if score
    ]


def select_topics(
    message: str, topics: Iterable[str] = FALLBACK_TOPICS, count: int = 3
) -> list[str]:
    """Select up to three relevant labels with deterministic rotation for ties."""
    available = tuple(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))
    if not available:
        available = FALLBACK_TOPICS
    count = min(max(count, 1), 3, len(available))
    relevant = matching_topics(message, available)
    offset = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:4], "big")
    rotated = available[offset:] + available[:offset]
    ordered = [*relevant, *(topic for topic in rotated if topic not in relevant)]
    return ordered[:count]


def format_topic_list(topics: Iterable[str]) -> str:
    """Join labels with standard comma-and-final-or list punctuation."""
    items = tuple(topics)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def topic_examples(topics: Iterable[str] = FALLBACK_TOPICS) -> list[list[str]]:
    """Create Gradio example-chip inputs from up to three curated topics."""
    available = tuple(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))
    if not available:
        available = FALLBACK_TOPICS
    return [[f"What do the notes say about {topic}?"] for topic in available[:3]]
