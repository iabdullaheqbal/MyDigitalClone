import gradio as gr

from clone.bootstrap import create_app
from clone.chat.session import ChatSession
from clone.service import CloneService
from clone.types import AnswerResult, Chunk, RetrievalHit, RetrievalMode
from clone.ui.formatting import format_evidence
from clone.ui.handlers import clear_conversation, handle_message
from clone.ui.layout import build_ui


class FakeLLM:
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        return "A grounded answer."


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]:
        chunk = Chunk("faq#0", "faq.md", "Grounded note.", 0)
        return [RetrievalHit(chunk, 1.0, 1.0, 0.8, 1, 1)]


class NoCallRetriever:
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]:
        raise AssertionError("UI construction must not retrieve")


def test_handler_updates_transcript_session_and_evidence() -> None:
    session = ChatSession()

    message, history, returned_session, evidence = handle_message(
        " A question ",
        [{"role": "assistant", "content": "Earlier answer"}],
        session,
        RetrievalMode.HYBRID_RRF.value,
        5,
        False,
        service=CloneService(FakeRetriever(), FakeLLM()),
    )

    assert message == ""
    assert history == [
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "A question"},
        {"role": "assistant", "content": "A grounded answer."},
    ]
    assert returned_session is session
    assert [turn.content for turn in session.history()] == [
        "A question",
        "A grounded answer.",
    ]
    assert evidence["retrieval_query"] == "A question"
    assert evidence["used_groq"] is True


def test_clear_creates_fresh_transient_session_state() -> None:
    messages, session, evidence = clear_conversation()

    assert messages == []
    assert session.history() == []
    assert evidence == {}


def test_ui_builds_gradio_blocks_without_loading_models_or_launching() -> None:
    service = CloneService(NoCallRetriever(), None)

    demo = build_ui(service, reranker_available=False)

    assert isinstance(demo, gr.Blocks)


def test_app_factory_builds_from_the_saved_index_without_launching() -> None:
    assert isinstance(create_app(), gr.Blocks)


def test_evidence_format_keeps_retrieval_and_timing_metadata() -> None:
    result = AnswerResult(
        answer="Answer",
        standalone_query="query",
        hits=[],
        mode=RetrievalMode.DENSE_ONLY,
        declined=False,
        used_llm=False,
        latency_ms={"retrieval": 2.5},
    )

    formatted = format_evidence(result)

    assert formatted["retrieval_mode"] == "Dense only"
    assert formatted["latency_ms"] == {"retrieval": 2.5}
    assert formatted["passages"] == []
