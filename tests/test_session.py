from clone.chat.session import ChatSession
from clone.types import Turn


def test_sessions_are_isolated_and_clear_only_their_own_history() -> None:
    first = ChatSession()
    second = ChatSession()
    first.record_exchange("Question", "Answer")

    assert first.history() == [Turn("user", "Question"), Turn("assistant", "Answer")]
    assert second.history() == []

    first.clear()
    assert first.history() == []
    assert second.history() == []


def test_history_returns_a_copy_and_empty_questions_are_rejected() -> None:
    session = ChatSession()
    history = session.history()
    history.append(Turn("user", "outside mutation"))

    assert session.history() == []

    try:
        session.record_exchange("  ", "Answer")
    except ValueError as error:
        assert str(error) == "Question must not be empty"
    else:
        raise AssertionError("Expected an empty question to be rejected")
