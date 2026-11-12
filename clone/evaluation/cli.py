"""Command-line entrypoint shared by evaluation scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clone.config import Settings
from clone.evaluation.dataset import load_evaluation_dataset
from clone.evaluation.report import report_to_json, write_report
from clone.evaluation.runner import build_retriever, run_retrieval_evaluation
from clone.types import RetrievalMode

ROOT = Path(__file__).resolve().parents[2]


def main(compare_modes: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "eval_set.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", action="append", choices=[mode.value for mode in RetrievalMode])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-drafts",
        action="store_true",
        help="Run draft cases marked for human review; results remain unvalidated.",
    )
    args = parser.parse_args()
    if compare_modes and args.mode:
        parser.error("--mode cannot be combined with the compare-modes entry point")

    settings = Settings.from_env()
    dataset = load_evaluation_dataset(args.dataset, allow_drafts=args.allow_drafts)
    modes = (
        tuple(RetrievalMode)
        if compare_modes
        else tuple(RetrievalMode(value) for value in args.mode)
        if args.mode
        else (RetrievalMode.HYBRID_RRF,)
    )
    report = run_retrieval_evaluation(
        dataset,
        build_retriever(settings),
        modes=modes,
        k=args.k,
        candidates=settings.candidates,
    )
    if args.output:
        write_report(args.output, report)
    else:
        sys.stdout.write(report_to_json(report))
