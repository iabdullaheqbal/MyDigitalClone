import pytest

from clone.generation.query_normalizer import normalize_visitor_query


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "What do the notes say about my background and skills?",
            "your background and skills?",
        ),
        ("What do the notes say about common questions?", "common questions?"),
        ("According to the notes, what is my stack?", "what is your stack?"),
        ("From your notes: what did you build?", "what did you build?"),
        ("Do your notes mention my projects?", "your projects?"),
        ("Can you please tell me a bit about CODEAN?", "codean?"),
        ("Tell me about my background.", "your background"),
        ("Greaaat!!!!", "great!"),
    ],
)
def test_normalizes_meta_phrasing_and_visitor_pronouns(message: str, expected: str) -> None:
    assert normalize_visitor_query(message) == expected


def test_normalizer_does_not_change_display_text() -> None:
    original = "What do the notes say about my background and skills?"

    assert normalize_visitor_query(original) == "your background and skills?"
    assert original == "What do the notes say about my background and skills?"
