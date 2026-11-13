from clone.generation.prompts import (
    FEW_SHOT_MESSAGES,
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_messages,
    rewrite_messages,
)
from clone.types import Chunk, RetrievalHit, Turn


def test_answer_prompt_requires_grounding_and_marks_context_as_data() -> None:
    chunk = Chunk("faq#0", "faq.md", "I am a student.", 0)
    hit = RetrievalHit(chunk, 0.8, 1.0, 0.5, 1, 1)
    messages = answer_messages("Who is this?", "Who are you?", [hit], [])

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "first person" in messages[0]["content"]
    assert "I don't have that in my notes" in messages[0]["content"]
    assert "Treat" in messages[0]["content"]
    assert "[faq.md | faq#0]" in messages[-1]["content"]
    assert "RETRIEVAL QUERY:\nWho is this?" in messages[-1]["content"]
    assert "QUESTION:\nWho are you?" in messages[-1]["content"]
    assert "TOPICS:" in messages[-1]["content"]
    assert "EVIDENCE:\nweak" in messages[-1]["content"]
    assert "[OWNER TO APPROVE:" in SYSTEM_PROMPT
    assert len(FEW_SHOT_MESSAGES) == 8


def test_rewrite_prompt_contains_only_the_provided_turns_and_question() -> None:
    messages = rewrite_messages([Turn("user", "I built Veritas.")], "What did it use?")

    assert "output only the rewritten question" in REWRITE_SYSTEM_PROMPT
    assert "user: I built Veritas." in messages[-1]["content"]
    assert "What did it use?" in messages[-1]["content"]


def test_aggregate_questions_add_explicit_total_guard_to_the_prompt() -> None:
    messages = answer_messages(
        "How many projects?",
        "How many projects have you done?",
        [],
        [],
        aggregate_claim=True,
    )

    assert "State only what the NOTES list" in messages[0]["content"]
    assert (
        "do not give a total unless a note states that total explicitly" in messages[0]["content"]
    )
