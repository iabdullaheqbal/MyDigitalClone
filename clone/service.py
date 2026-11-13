"""Application orchestration for routed, session-aware answers."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from clone.chat.session import ChatSession
from clone.generation.answerer import Answerer
from clone.generation.guard import assess_evidence
from clone.generation.intent import Intent, is_about_me_question
from clone.generation.llm_client import ChatClient
from clone.generation.query_rewriter import rewrite
from clone.generation.responses import render_response, response_category
from clone.generation.topics import FALLBACK_TOPICS
from clone.guardrails.input_checks import MAX_SESSION_MESSAGES, check_input
from clone.guardrails.privacy import redact_names, redact_retrieval_hits
from clone.types import AnswerResult, RetrievalHit, RetrievalMode

logger = logging.getLogger(__name__)
_BYPASS_INTENTS = {
    Intent.GREETING,
    Intent.THANKS,
    Intent.ACKNOWLEDGEMENT,
    Intent.GOODBYE,
    Intent.IDENTITY,
    Intent.SENSITIVE_REQUEST,
    Intent.INJECTION,
    Intent.ABUSE,
    Intent.GIBBERISH_OR_EMPTY,
}


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
    """Coordinate intent routing, evidence grading, generation, and session updates."""

    def __init__(
        self,
        retriever: Retriever,
        llm: ChatClient | None,
        *,
        strong_cosine: float = 0.45,
        weak_cosine: float = 0.20,
        candidates: int = 20,
        history_turns: int = 6,
        topics: tuple[str, ...] = FALLBACK_TOPICS,
        show_supervisor_name: bool = False,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        if history_turns < 0:
            raise ValueError("history_turns must not be negative")
        if not -1.0 <= weak_cosine <= strong_cosine <= 1.0:
            raise ValueError("Evidence thresholds must satisfy -1 <= weak <= strong <= 1")
        self._retriever = retriever
        self._llm = llm
        self._answerer = Answerer(llm, topics)
        self._candidates = candidates
        self._history_turns = history_turns
        self.topics = topics or FALLBACK_TOPICS
        self._show_supervisor_name = show_supervisor_name
        self._strong_cosine = strong_cosine
        self._weak_cosine = weak_cosine

    def answer(
        self,
        question: str,
        session: ChatSession,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> AnswerResult:
        """Answer one message, routing special intents before rewrite or retrieval."""
        normalized_question = question.strip()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        selected_mode = RetrievalMode(mode)
        started = perf_counter()
        input_check = check_input(question)
        flags = list(input_check.flags)
        intent = input_check.intent_override or input_check.intent

        if not normalized_question:
            return self._template_result(
                "",
                session,
                selected_mode,
                "empty",
                Intent.GIBBERISH_OR_EMPTY,
                flags,
                started,
                retain=False,
            )
        if "too_long" in flags:
            return self._template_result(
                normalized_question,
                session,
                selected_mode,
                "too_long",
                Intent.OUT_OF_SCOPE,
                flags,
                started,
                retain=False,
            )
        if session.user_message_count >= MAX_SESSION_MESSAGES:
            return self._template_result(
                normalized_question,
                session,
                selected_mode,
                "session_limit",
                Intent.OUT_OF_SCOPE,
                [*flags, "session_limit"],
                started,
                retain=False,
            )
        if intent in _BYPASS_INTENTS:
            return self._template_result(
                normalized_question,
                session,
                selected_mode,
                response_category(intent),
                intent,
                flags,
                started,
            )

        history = session.history()
        context_history = history[-self._history_turns :] if self._history_turns else []
        standalone_query = normalized_question
        retrieval_started = perf_counter()
        hits = self._retriever.retrieve(
            standalone_query,
            selected_mode,
            k=top_k,
            candidates=self._candidates,
            use_reranker=use_reranker,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        evidence_level = assess_evidence(
            hits,
            strong_cosine=self._strong_cosine,
            weak_cosine=self._weak_cosine,
        )
        hits, third_party_names = redact_retrieval_hits(hits, show_names=self._show_supervisor_name)
        if third_party_names:
            context_history = [
                replace(turn, content=redact_names(turn.content, third_party_names)[0])
                for turn in context_history
            ]
        rewrite_ms = 0.0

        if evidence_level != "none" and self._llm is not None and context_history:
            rewrite_started = perf_counter()
            rewritten_query = rewrite(context_history, normalized_question, self._llm)
            rewrite_ms = (perf_counter() - rewrite_started) * 1000
            if rewritten_query != normalized_question:
                retrieval_started = perf_counter()
                rewritten_hits = self._retriever.retrieve(
                    rewritten_query,
                    selected_mode,
                    k=top_k,
                    candidates=self._candidates,
                    use_reranker=use_reranker,
                )
                retrieval_ms += (perf_counter() - retrieval_started) * 1000
                rewritten_evidence = assess_evidence(
                    rewritten_hits,
                    strong_cosine=self._strong_cosine,
                    weak_cosine=self._weak_cosine,
                )
                if rewritten_evidence != "none":
                    standalone_query = rewritten_query
                    rewritten_hits, rewritten_names = redact_retrieval_hits(
                        rewritten_hits, show_names=self._show_supervisor_name
                    )
                    third_party_names = tuple(dict.fromkeys((*third_party_names, *rewritten_names)))
                    if third_party_names and not self._show_supervisor_name:
                        rewritten_hits = [
                            replace(
                                hit,
                                chunk=replace(
                                    hit.chunk,
                                    text=redact_names(hit.chunk.text, third_party_names)[0],
                                ),
                            )
                            for hit in rewritten_hits
                        ]
                    hits = rewritten_hits
                    if third_party_names:
                        context_history = [
                            replace(turn, content=redact_names(turn.content, third_party_names)[0])
                            for turn in context_history
                        ]
                    evidence_level = rewritten_evidence

        if evidence_level == "none":
            intent = (
                Intent.NOT_IN_NOTES
                if is_about_me_question(normalized_question)
                else Intent.OUT_OF_SCOPE
            )
            category = "not_in_notes" if intent is Intent.NOT_IN_NOTES else "out_of_scope"
            flags.append("no_evidence")
            answer = render_response(category, normalized_question, self.topics)
            session.record_exchange(
                normalized_question,
                answer,
                retain_in_history=intent is Intent.NOT_IN_NOTES,
            )
            total_ms = (perf_counter() - started) * 1000
            logger.info("Message routed: intent=%s flags=%s", intent.value, tuple(flags))
            return AnswerResult(
                answer=answer,
                standalone_query=standalone_query,
                hits=hits,
                mode=selected_mode,
                declined=True,
                used_llm=False,
                latency_ms={
                    "query_rewrite": rewrite_ms,
                    "retrieval": retrieval_ms,
                    "generation": 0.0,
                    "total": total_ms,
                },
                intent=intent.value,
                evidence_level=evidence_level,
                guardrail_flags=tuple(flags),
                not_found_in_notes=intent is Intent.NOT_IN_NOTES,
            )

        intent = Intent.ABOUT_ME
        generation_started = perf_counter()
        generated = self._answerer.answer(
            standalone_query,
            normalized_question,
            hits,
            context_history,
            evidence_level,
            third_party_names,
        )
        generation_ms = (perf_counter() - generation_started) * 1000
        session.record_exchange(normalized_question, generated.answer)
        all_flags = tuple(dict.fromkeys([*flags, *generated.guardrail_flags]))
        total_ms = (perf_counter() - started) * 1000
        logger.info("Message routed: intent=%s flags=%s", intent.value, all_flags)

        return AnswerResult(
            answer=generated.answer,
            standalone_query=standalone_query,
            hits=hits,
            mode=selected_mode,
            declined=not generated.used_llm,
            used_llm=generated.used_llm,
            latency_ms={
                "query_rewrite": rewrite_ms,
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": total_ms,
            },
            error=generated.error,
            intent=intent.value,
            evidence_level=evidence_level,
            guardrail_flags=all_flags,
            not_found_in_notes=False,
        )

    def _template_result(
        self,
        question: str,
        session: ChatSession,
        mode: RetrievalMode,
        category: str,
        intent: Intent,
        flags: list[str],
        started: float,
        *,
        retain: bool = False,
    ) -> AnswerResult:
        previous_answer = next(
            (turn.content for turn in reversed(session.history()) if turn.role == "assistant"),
            None,
        )
        answer = render_response(
            category,
            question,
            self.topics,
            previous_answer=previous_answer,
        )
        if question:
            session.record_exchange(question, answer, retain_in_history=retain)
        all_flags = tuple(dict.fromkeys(flags))
        logger.info("Message routed: intent=%s flags=%s", intent.value, all_flags)
        return AnswerResult(
            answer=answer,
            standalone_query=question,
            hits=[],
            mode=mode,
            declined=True,
            used_llm=False,
            latency_ms={
                "query_rewrite": 0.0,
                "retrieval": 0.0,
                "generation": 0.0,
                "total": (perf_counter() - started) * 1000,
            },
            intent=intent.value,
            evidence_level="none",
            guardrail_flags=all_flags,
            not_found_in_notes=category == "not_in_notes",
        )
