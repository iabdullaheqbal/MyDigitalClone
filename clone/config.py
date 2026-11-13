"""Validated application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_int(name: str, default: int, minimum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _read_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    groq_api_key: str | None
    groq_model: str
    groq_base_url: str
    embed_model: str
    chunk_words: int
    overlap_words: int
    rrf_k: int
    candidates: int
    strong_cosine: float
    weak_cosine: float
    history_turns: int
    show_supervisor_name: bool
    data_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        """Create validated settings without requiring an LLM key."""
        api_key = os.getenv("GROQ_API_KEY", "").strip() or None
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")
        embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2").strip()
        if not model or not base_url or not embed_model:
            raise ValueError("GROQ_MODEL, GROQ_BASE_URL, and EMBED_MODEL must not be empty")
        strong_cosine = _read_float("STRONG_COSINE", 0.45, -1.0, 1.0)
        weak_cosine = _read_float("WEAK_COSINE", 0.20, -1.0, 1.0)
        if strong_cosine < weak_cosine:
            raise ValueError("STRONG_COSINE must be greater than or equal to WEAK_COSINE")

        return cls(
            groq_api_key=api_key,
            groq_model=model,
            groq_base_url=base_url,
            embed_model=embed_model,
            chunk_words=_read_int("CHUNK_WORDS", 250, 1),
            overlap_words=_read_int("OVERLAP_WORDS", 40, 0),
            rrf_k=_read_int("RRF_K", 60, 1),
            candidates=_read_int("CANDIDATES", 20, 1),
            strong_cosine=strong_cosine,
            weak_cosine=weak_cosine,
            history_turns=_read_int("HISTORY_TURNS", 6, 0),
            show_supervisor_name=_read_bool("SHOW_SUPERVISOR_NAME", False),
            data_dir=Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser(),
            artifacts_dir=Path(
                os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
            ).expanduser(),
        )
