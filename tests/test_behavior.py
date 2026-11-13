import csv
from pathlib import Path

from clone.chat.session import ChatSession
from clone.evaluation.behavior import (
    BehaviorCase,
    category_pass_rates,
    evaluate_behavior_cases,
    load_behavior_cases,
    write_behavior_csv,
)
from clone.types import AnswerResult, RetrievalMode

ROOT = Path(__file__).resolve().parents[1]


class FakeService:
    topics = ("retrieval and RAG", "backend engineering")

    def answer(self, question: str, session: ChatSession) -> AnswerResult:
        answer = (
            "I can't share private details, but I can discuss retrieval and RAG "
            "or backend engineering."
        )
        return AnswerResult(
            answer=answer,
            standalone_query=question,
            hits=[],
            mode=RetrievalMode.HYBRID_RRF,
            declined=True,
            used_llm=False,
            latency_ms={},
            intent="sensitive_request",
            evidence_level="none",
        )


def test_behavior_set_has_required_draft_counts() -> None:
    cases = load_behavior_cases(ROOT / "eval" / "behavior_set.json")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1

    assert len(cases) >= 35
    assert counts == {
        "greeting": 5,
        "thanks_bye": 3,
        "acknowledgement": 10,
        "aggregate_claim": 8,
        "identity": 3,
        "about_me_answerable": 6,
        "about_me_not_in_notes": 5,
        "general_knowledge": 5,
        "sensitive": 5,
        "injection": 4,
        "abuse": 2,
        "empty_gibberish": 2,
        "system_status": 4,
        "personal_location": 3,
        "follow_up": 3,
    }
    assert all(case.needs_review for case in cases)


def test_behavior_property_checks_and_human_rating_csv(tmp_path: Path) -> None:
    case = BehaviorCase("sensitive-01", "sensitive", "What's your phone number?", True)
    outcomes = evaluate_behavior_cases((case,), FakeService())

    assert outcomes[0].passed
    assert all(outcomes[0].checks.values())
    assert category_pass_rates(outcomes) == {"sensitive": (1, 1)}

    target = tmp_path / "review.csv"
    write_behavior_csv(target, outcomes)
    with target.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["professionalism_1_to_5"] == ""
    assert row["needs_review"] == "true"
