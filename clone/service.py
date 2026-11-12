"""Application orchestration for session-aware grounded answers."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

from clone.chat.session import ChatSession
from clone.generation.answerer import Answerer
from clone.generation.llm_client import ChatClient
from clone.generation.query_rewriter import rewrite
from clone.types import AnswerResult, RetrievalHit, RetrievalMode


class Retriever(Protocol):
    """Retrieval interface consumed by the application service."""

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]: ...


class CloneService:
    """Coordinate rewriting, retrieval, generation, and session updates."""

    def __init__(
        self,
        retriever: Retriever,
        llm: ChatClient | None,
        *,
        dense_min: float = 0.20,
        candidates: int = 20,
        history_turns: int = 6,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        if history_turns < 0:
            raise ValueError("history_turns must not be negative")
        self._retriever = retriever
        self._llm = llm
        self._answerer = Answerer(llm, dense_min=dense_min)
        self._candidates = candidates
        self._history_turns = history_turns

    def answer(
        self,
        question: str,
        session: ChatSession,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> AnswerResult:
        """Answer one question and append the result to that session."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        selected_mode = RetrievalMode(mode)
        history = session.history()
        context_history = history[-self._history_turns :] if self._history_turns else []

        started = perf_counter()
        standalone_query = (
            rewrite(context_history, normalized_question, self._llm)
            if self._llm is not None and context_history
            else normalized_question
        )
        rewrite_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        hits = self._retriever.retrieve(
            standalone_query,
            selected_mode,
            k=top_k,
            candidates=self._candidates,
            use_reranker=use_reranker,
        )
        retrieval_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        answer, used_llm, declined, error = self._answerer.answer(
            standalone_query,
            normalized_question,
            hits,
            context_history,
        )
        generation_ms = (perf_counter() - started) * 1000
        session.record_exchange(normalized_question, answer)

        return AnswerResult(
            answer=answer,
            standalone_query=standalone_query,
            hits=hits,
            mode=selected_mode,
            declined=declined,
            used_llm=used_llm,
            latency_ms={
                "query_rewrite": rewrite_ms,
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": rewrite_ms + retrieval_ms + generation_ms,
            },
            error=error,
        )
