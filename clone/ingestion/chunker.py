"""Deterministic paragraph-aware word chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    continuation: bool = False


def _units(paragraphs: list[str], chunk_words: int, overlap_words: int) -> list[_Unit]:
    units = []
    step = chunk_words - overlap_words
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= chunk_words:
            units.append(_Unit(paragraph))
            continue
        for start in range(0, len(words), step):
            units.append(
                _Unit(
                    text=" ".join(words[start : start + chunk_words]),
                    continuation=start > 0,
                )
            )
            if start + chunk_words >= len(words):
                break
    return units


def _tail_words(paragraphs: list[str], count: int) -> list[str]:
    if count == 0:
        return []
    return " ".join(paragraphs).split()[-count:]


def chunk_text(
    text: str,
    chunk_words: int = 250,
    overlap_words: int = 40,
) -> list[str]:
    """Pack paragraphs into bounded chunks with deterministic word overlap."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    chunks = []
    current: list[str] = []
    current_words = 0
    for unit in _units(paragraphs, chunk_words, overlap_words):
        unit_size = len(unit.text.split())
        if current and current_words + unit_size > chunk_words:
            chunks.append("\n\n".join(current))
            carry_size = 0 if unit.continuation else min(overlap_words, chunk_words - unit_size)
            carry = _tail_words(current, carry_size)
            current = [" ".join(carry)] if carry else []
            current_words = len(carry)

        current.append(unit.text)
        current_words += unit_size

    if current:
        chunks.append("\n\n".join(current))
    return chunks
