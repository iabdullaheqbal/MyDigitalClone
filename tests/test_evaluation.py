import json
from pathlib import Path

import pytest

from clone.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
    load_evaluation_dataset,
    load_multiturn_dataset,
)
from clone.evaluation.manual_scoring import (
    load_manual_scores,
    scoring_template,
    summarize_manual_scores,
)
from clone.evaluation.metrics import score_retrieval
from clone.evaluation.report import report_to_dict, write_report
from clone.evaluation.runner import (
    run_multiturn_retrieval_evaluation,
    run_retrieval_evaluation,
)
from clone.types import Chunk, RetrievalHit, RetrievalMode

ROOT = Path(__file__).resolve().parents[1]


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[tuple[str, RetrievalMode]] = []

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]:
        self.queries.append((query, mode))
        chunks = [
            Chunk("a#0", "a.md", "relevant", 0),
            Chunk("a#1", "a.md", "duplicate source", 1),
            Chunk("b#0", "b.md", "other", 2),
        ]
        return [
            RetrievalHit(chunk, 1.0, 1.0, 0.8, 1, 1)
            for chunk in chunks[:k]
        ]


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="fixture",
        review_status="reviewed",
        cases=(EvaluationCase("case-1", "question", ("a.md",), "test", False),),
        human_review_required=False,
    )


def test_draft_datasets_require_explicit_permission() -> None:
    path = ROOT / "eval" / "eval_set.json"

    with pytest.raises(ValueError, match="requires human review"):
        load_evaluation_dataset(path)

    dataset = load_evaluation_dataset(path, allow_drafts=True)
    assert len(dataset.cases) == 12
    assert dataset.human_review_required
    assert all(case.human_review_required for case in dataset.cases)

    multiturn = load_multiturn_dataset(
        ROOT / "eval" / "eval_multiturn.json",
        allow_drafts=True,
    )
    assert len(multiturn.cases) == 1
    assert multiturn.cases[0].history
    assert multiturn.human_review_required


def test_retrieval_metrics_deduplicate_sources_and_use_fixed_cutoff() -> None:
    scores = score_retrieval(["a.md", "a.md", "b.md"], ["a.md", "b.md"], k=3)

    assert scores.precision_at_k == pytest.approx(2 / 3)
    assert scores.recall_at_k == 1.0
    assert scores.reciprocal_rank_at_k == 1.0


def test_runner_compares_modes_and_keeps_draft_status() -> None:
    report = run_retrieval_evaluation(_dataset(), FakeRetriever(), k=2)

    assert [mode.mode for mode in report.modes] == [mode.value for mode in RetrievalMode]
    assert all(mode.mean_precision_at_k == pytest.approx(0.5) for mode in report.modes)
    assert all(mode.mean_recall_at_k == 1.0 for mode in report.modes)
    assert report.human_review_required is False
    assert report.modes[0].queries[0].retrieved_sources == ("a.md", "a.md")


def test_multiturn_runner_uses_injected_rewriter_without_generation_calls() -> None:
    dataset = load_multiturn_dataset(
        ROOT / "eval" / "eval_multiturn.json",
        allow_drafts=True,
    )
    retriever = FakeRetriever()
    rewrite_calls: list[tuple[str, str]] = []

    def fake_rewriter(history, question):
        rewrite_calls.append((history[-1].content, question))
        return "standalone query"

    report = run_multiturn_retrieval_evaluation(
        dataset,
        retriever,
        fake_rewriter,
        k=2,
    )

    assert len(rewrite_calls) == 1
    assert retriever.queries == [
        ("standalone query", mode) for mode in RetrievalMode
    ]
    assert report.human_review_required
    assert report.modes[0].queries[0].standalone_query == "standalone query"
    assert (
        report.modes[0].queries[0].expected_standalone_query
        == dataset.cases[0].expected_standalone_query
    )


def test_report_serialization_preserves_metrics_and_review_gate(tmp_path) -> None:
    report = run_retrieval_evaluation(_dataset(), FakeRetriever(), modes=[RetrievalMode.BM25_ONLY])
    output = tmp_path / "nested" / "report.json"

    write_report(output, report)
    parsed = json.loads(output.read_text(encoding="utf-8"))

    assert parsed == report_to_dict(report)
    assert parsed["modes"][0]["mode"] == RetrievalMode.BM25_ONLY.value


def test_manual_score_template_and_validation(tmp_path) -> None:
    template = scoring_template(["case-1"])
    row = template["scores"][0]
    assert row["groundedness"] is None

    path = tmp_path / "scores.json"
    template["review_status"] = "reviewed"
    template["scores"] = [
        {
            "case_id": "case-1",
            "groundedness": 4,
            "completeness": 3,
            "citation_accuracy": 5,
            "notes": "Reviewed",
        }
    ]
    path.write_text(json.dumps(template), encoding="utf-8")
    scores = load_manual_scores(path)

    assert summarize_manual_scores(scores) == {
        "groundedness": 4.0,
        "completeness": 3.0,
        "citation_accuracy": 5.0,
    }

    template["scores"][0]["groundedness"] = True
    path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="integers from 1 through 5"):
        load_manual_scores(path)
