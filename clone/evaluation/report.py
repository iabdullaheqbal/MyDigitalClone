"""Serialize and persist evaluation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from clone.evaluation.runner import EvaluationReport


def report_to_dict(report: EvaluationReport) -> dict[str, object]:
    """Convert the typed report into JSON-compatible values."""
    return json.loads(json.dumps(asdict(report), ensure_ascii=False))


def report_to_json(report: EvaluationReport) -> str:
    """Return a stable, readable JSON representation."""
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=2) + "\n"


def write_report(path: Path, report: EvaluationReport) -> None:
    """Write a report to the user-selected output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report), encoding="utf-8")
