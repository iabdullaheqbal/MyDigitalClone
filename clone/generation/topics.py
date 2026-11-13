"""Topic suggestions derived from indexed note titles, with safe static defaults."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from clone.ingestion.loader import Document

FALLBACK_TOPICS = (
    "background and education",
    "skills and work style",
    "backend engineering and infrastructure",
    "machine-learning pipelines and evaluation",
    "retrieval, RAG, and reranking",
    "cybersecurity and digital forensics",
    "AI products and project work",
    "local LLMs and automation",
)
_TOPIC_GROUPS = (
    ("background and education", re.compile(r"\bbio\b|\bcv\b|\beducation\b", re.I)),
    (
        "skills and work style",
        re.compile(r"cv[_ -]?skills|work[_ -]?style|how[_ -]?i[_ -]?work", re.I),
    ),
    (
        "backend engineering and infrastructure",
        re.compile(
            r"backend|infrastructure|api|database|cloud(?:\s+computing|[_ -]?infrastructure)",
            re.I,
        ),
    ),
    (
        "machine-learning pipelines and evaluation",
        re.compile(r"machine[_ -]?learning|ml[_ -]?pipelines|evaluation|transformer", re.I),
    ),
    ("retrieval, RAG, and reranking", re.compile(r"retrieval|\brag\b|rerank|bm25", re.I)),
    (
        "cybersecurity and digital forensics",
        re.compile(r"security|cyber|forensic|phishing|intrusion", re.I),
    ),
    (
        "AI products and project work",
        re.compile(r"project|product|codean|resona|internship", re.I),
    ),
    (
        "local LLMs and automation",
        re.compile(r"local[_ -]?llm|automation|ollama|n8n|language[_ -]?model", re.I),
    ),
)


def derive_topics(documents: Iterable[Document]) -> list[str]:
    """Map filenames/headings to concise topic groups; keep raw titles as fallback."""
    documents = tuple(documents)
    title_text: list[str] = []
    headings: list[str] = []
    for document in documents:
        title_text.append(Path(document.name).stem.replace("_", " ").replace("-", " "))
        headings.extend(
            re.sub(r"\s+", " ", match.group(1).strip(" #"))
            for match in re.finditer(r"(?m)^#{1,3}\s+(.+?)\s*#*\s*$", document.text)
        )
    metadata_text = " ".join((*title_text, *headings))
    grouped = [label for label, pattern in _TOPIC_GROUPS if pattern.search(metadata_text)]
    if grouped:
        return grouped

    raw_topics = list(dict.fromkeys((*headings, *title_text)))
    return [topic[:80] for topic in raw_topics if topic.strip()] or list(FALLBACK_TOPICS)


def select_topics(
    message: str, topics: Iterable[str] = FALLBACK_TOPICS, count: int = 2
) -> list[str]:
    """Choose relevant topics deterministically, using overlap then a stable hash."""
    available = tuple(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))
    if not available:
        available = FALLBACK_TOPICS
    count = min(max(count, 1), len(available))
    query_words = set(re.findall(r"[a-z0-9]+", message.casefold()))
    scored = [
        (
            sum(word in query_words for word in re.findall(r"[a-z0-9]+", topic.casefold())),
            index,
            topic,
        )
        for index, topic in enumerate(available)
    ]
    if any(score for score, _, _ in scored):
        return [
            topic for _, _, topic in sorted(scored, key=lambda item: (-item[0], item[1]))[:count]
        ]
    offset = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:4], "big")
    return [available[(offset + index) % len(available)] for index in range(count)]


def topic_examples(topics: Iterable[str] = FALLBACK_TOPICS) -> list[list[str]]:
    """Create Gradio example-chip inputs from the available note topics."""
    available = tuple(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))
    if not available:
        available = FALLBACK_TOPICS
    return [[f"What do the notes say about {topic}?"] for topic in available[:4]]
