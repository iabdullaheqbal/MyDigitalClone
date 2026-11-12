"""Create and validate human answer-quality scoring forms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

SCORE_FIELDS = ("groundedness", "completeness", "citation_accuracy")


@dataclass(frozen=True, slots=True)
class ManualScore:
    """One reviewer assessment of a generated answer."""

    case_id: str
    groundedness: int
    completeness: int
    citation_accuracy: int
    notes: str


def scoring_template(case_ids: list[str]) -> dict[str, object]:
    """Create blank 1-to-5 review rows without inventing answer scores."""
    if not case_ids or any(not case_id.strip() for case_id in case_ids):
        raise ValueError("At least one non-empty case id is required")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case ids must be unique")
    return {
        "review_status": "pending",
        "rubric_scale": "1 (poor) to 5 (excellent)",
        "scores": [
            {
                "case_id": case_id,
                "groundedness": None,
                "completeness": None,
                "citation_accuracy": None,
                "notes": "",
            }
            for case_id in case_ids
        ],
    }


def load_manual_scores(path: Path) -> tuple[ManualScore, ...]:
    """Load completed reviewer scores, rejecting incomplete or invalid forms."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("review_status") != "reviewed":
        raise ValueError("Manual scores must be marked review_status='reviewed'")
    rows = payload.get("scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manual scores must contain a non-empty scores list")
    scores = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each manual score must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each score must have a unique, non-empty case_id")
        case_id = case_id.strip()
        if case_id in seen:
            raise ValueError("Each score must have a unique, non-empty case_id")
        seen.add(case_id)
        groundedness = _read_score(row.get("groundedness"))
        completeness = _read_score(row.get("completeness"))
        citation_accuracy = _read_score(row.get("citation_accuracy"))
        notes = row.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("Manual score notes must be text")
        scores.append(
            ManualScore(
                case_id,
                groundedness,
                completeness,
                citation_accuracy,
                notes,
            )
        )
    return tuple(scores)


def _read_score(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError("Manual scores must be integers from 1 through 5")
    return value


def summarize_manual_scores(scores: tuple[ManualScore, ...]) -> dict[str, float]:
    """Average each human-scored answer-quality dimension."""
    if not scores:
        raise ValueError("At least one manual score is required")
    return {
        field: mean(getattr(score, field) for score in scores)
        for field in SCORE_FIELDS
    }
