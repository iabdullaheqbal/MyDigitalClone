"""Run draft behavior cases and export a human professionalism-rating sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from clone.config import Settings
from clone.evaluation.behavior import (
    category_pass_rates,
    evaluate_behavior_cases,
    load_behavior_cases,
    write_behavior_csv,
)
from clone.evaluation.runner import build_retriever
from clone.generation.llm_client import GroqClient
from clone.indexing.store import load_faq_questions, load_topics
from clone.service import CloneService

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Run the clearly labeled, unreviewed behavior dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", action="store_true", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "behavior_set.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "behavior_review.csv",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    cases = load_behavior_cases(args.dataset)
    llm = (
        GroqClient(settings.groq_api_key, settings.groq_model, settings.groq_base_url)
        if settings.groq_api_key
        else None
    )
    service = CloneService(
        build_retriever(settings),
        llm,
        strong_cosine=settings.strong_cosine,
        weak_cosine=settings.weak_cosine,
        candidates=settings.candidates,
        history_turns=settings.history_turns,
        topics=load_topics(settings.artifacts_dir),
        faq_questions=load_faq_questions(settings.artifacts_dir),
    )
    outcomes = evaluate_behavior_cases(cases, service)
    print("Draft behavior evaluation; needs human review and is not validated.")
    for category, (passed, total) in category_pass_rates(outcomes).items():
        print(f"{category}: {passed}/{total} ({passed / total:.0%})")
    write_behavior_csv(args.output, outcomes)
    print(f"Human review sheet: {args.output}")
