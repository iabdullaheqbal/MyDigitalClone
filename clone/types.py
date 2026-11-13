"""Shared data types for ingestion, retrieval, conversations, and answers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class RetrievalMode(StrEnum):
    """Supported retrieval strategies."""

    BM25_ONLY = "BM25 only"
    DENSE_ONLY = "Dense only"
    HYBRID_RRF = "Hybrid (RRF)"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A source passage with a stable identifier and position."""

    id: str
    source: str
    text: str
    index: int


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A chunk and the scores/ranks produced by the retrieval pipeline."""

    chunk: Chunk
    score: float
    bm25_score: float
    dense_score: float
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """One user or assistant message in the current browser session."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Answer and evidence returned for a single user question."""

    answer: str
    standalone_query: str
    hits: list[RetrievalHit]
    mode: RetrievalMode
    declined: bool
    used_llm: bool
    latency_ms: dict[str, float]
    error: str | None = None
    intent: str = "about_me_or_other"
    evidence_level: str = "none"
    guardrail_flags: tuple[str, ...] = ()
    not_found_in_notes: bool = False
