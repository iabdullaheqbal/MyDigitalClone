"""Load reviewed retrieval and multi-turn evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clone.types import Turn


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One source-relevance judgment for a standalone question."""

    id: str
    question: str
    relevant_sources: tuple[str, ...]
    category: str
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned set of single-turn relevance judgments."""

    name: str
    review_status: str
    cases: tuple[EvaluationCase, ...]
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class MultiTurnCase:
    """A follow-up question with prior conversation and source judgments."""

    id: str
    history: tuple[Turn, ...]
    question: str
    expected_standalone_query: str
    relevant_sources: tuple[str, ...]
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class MultiTurnDataset:
    """A versioned set of multi-turn query-rewrite cases."""

    name: str
    review_status: str
    cases: tuple[MultiTurnCase, ...]
    human_review_required: bool


def _read_dataset(path: Path, expected_key: str) -> tuple[dict[str, object], list[object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation dataset must be a JSON object")
    if type(payload.get("version")) is not int or payload["version"] != 1:
        raise ValueError("Evaluation dataset version must be 1")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Evaluation dataset name must not be empty")
    status = payload.get("review_status")
    if not isinstance(status, str) or status not in {"draft", "reviewed"}:
        raise ValueError("review_status must be 'draft' or 'reviewed'")
    raw_cases = payload.get(expected_key)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Evaluation dataset must contain a non-empty {expected_key} list")
    return payload, raw_cases


def _read_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _read_sources(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("relevant_sources must be a non-empty list")
    sources = tuple(_read_string(source, "relevant source") for source in value)
    if len(set(sources)) != len(sources):
        raise ValueError("relevant_sources must not contain duplicates")
    return sources


def _read_review_marker(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("human_review_required must be a boolean")
    return value


def _check_review_status(
    status: str,
    item_markers: tuple[bool, ...],
    allow_drafts: bool,
) -> bool:
    requires_review = status == "draft" or any(item_markers)
    if status == "reviewed" and any(item_markers):
        raise ValueError("Reviewed datasets cannot contain items requiring human review")
    if requires_review and not allow_drafts:
        raise ValueError(
            "Evaluation data requires human review; explicitly allow drafts to proceed"
        )
    return requires_review


def load_evaluation_dataset(
    path: Path,
    *,
    allow_drafts: bool = False,
) -> EvaluationDataset:
    """Load source relevance cases, rejecting unreviewed data by default."""
    payload, raw_cases = _read_dataset(path, "cases")
    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Each evaluation case must be a JSON object")
        case_id = _read_string(raw_case.get("id"), "case id")
        question = _read_string(raw_case.get("question"), "question")
        category = _read_string(raw_case.get("category"), "category")
        sources = _read_sources(raw_case.get("relevant_sources"))
        review_required = _read_review_marker(raw_case.get("human_review_required"))
        cases.append(
            EvaluationCase(case_id, question, sources, category, review_required)
        )

    status = str(payload["review_status"])
    requires_review = _check_review_status(
        status,
        tuple(case.human_review_required for case in cases),
        allow_drafts,
    )
    return EvaluationDataset(
        name=str(payload["name"]),
        review_status=status,
        cases=tuple(cases),
        human_review_required=requires_review,
    )


def load_multiturn_dataset(
    path: Path,
    *,
    allow_drafts: bool = False,
) -> MultiTurnDataset:
    """Load multi-turn rewrite cases, rejecting unreviewed data by default."""
    payload, raw_cases = _read_dataset(path, "cases")
    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Each multi-turn case must be a JSON object")
        case_id = _read_string(raw_case.get("id"), "case id")
        raw_history = raw_case.get("history")
        if not isinstance(raw_history, list) or not raw_history:
            raise ValueError("history must be a non-empty list")
        history = []
        for raw_turn in raw_history:
            if not isinstance(raw_turn, dict):
                raise ValueError("Each history turn must be a JSON object")
            raw_role = raw_turn.get("role")
            if raw_role == "user":
                role = "user"
            elif raw_role == "assistant":
                role = "assistant"
            else:
                raise ValueError("History roles must be 'user' or 'assistant'")
            history.append(Turn(role, _read_string(raw_turn.get("content"), "turn content")))
        cases.append(
            MultiTurnCase(
                id=case_id,
                history=tuple(history),
                question=_read_string(raw_case.get("question"), "question"),
                expected_standalone_query=_read_string(
                    raw_case.get("expected_standalone_query"),
                    "expected_standalone_query",
                ),
                relevant_sources=_read_sources(raw_case.get("relevant_sources")),
                human_review_required=_read_review_marker(
                    raw_case.get("human_review_required")
                ),
            )
        )

    status = str(payload["review_status"])
    requires_review = _check_review_status(
        status,
        tuple(case.human_review_required for case in cases),
        allow_drafts,
    )
    return MultiTurnDataset(
        name=str(payload["name"]),
        review_status=status,
        cases=tuple(cases),
        human_review_required=requires_review,
    )
