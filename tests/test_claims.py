import pytest

from clone.guardrails.claims import is_aggregate_claim


@pytest.mark.parametrize(
    "question",
    [
        "How many projects have you done?",
        "How much experience do you have?",
        "What is the total number?",
        "List all your projects.",
        "Have you covered every topic?",
        "Which is your best project?",
        "What is the most important skill?",
        "Have you ever worked with this?",
        "Are these the only projects?",
        "What was your first project?",
        "What was your last project?",
    ],
)
def test_aggregate_claim_detector_recognizes_totals_and_superlatives(question: str) -> None:
    assert is_aggregate_claim(question)


def test_aggregate_claim_detector_leaves_specific_questions_unmarked() -> None:
    assert not is_aggregate_claim("Tell me about CODEAN.")
