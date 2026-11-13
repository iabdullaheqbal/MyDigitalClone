"""Application orchestration for routed, session-aware answers."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from clone.chat.session import ChatSession
from clone.generation.answer_cache import AnswerCache
from clone.generation.answerer import Answerer
from clone.generation.faq import FALLBACK_FAQ_QUESTIONS
from clone.generation.guard import assess_evidence
from clone.generation.intent import Intent, is_about_me_question
from clone.generation.llm_client import ChatClient
from clone.generation.query_normalizer import normalize_visitor_query
from clone.generation.query_rewriter import normalize_query, rewrite, should_rewrite
from clone.generation.responses import render_response, response_category, system_status_text
from clone.generation.topics import FALLBACK_TOPICS
from clone.guardrails.input_checks import MAX_SESSION_MESSAGES, check_input
from clone.guardrails.privacy import redact_names, redact_retrieval_hits
from clone.persona import PERSONA_VERSION
from clone.types import AnswerReason, AnswerResult, RetrievalHit, RetrievalMode, Turn

logger = logging.getLogger(__name__)
_BYPASS_INTENTS = {
    Intent.GREETING,
    Intent.THANKS,
    Intent.ACKNOWLEDGEMENT,
    Intent.GOODBYE,
    Intent.IDENTITY,
    Intent.STATUS,
    Intent.PERSONAL_LOCATION,
    Intent.LIST_FAQ,
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
        history_turns: int = 2,
        topics: tuple[str, ...] = FALLBACK_TOPICS,
        faq_questions: tuple[str, ...] = FALLBACK_FAQ_QUESTIONS,
        show_supervisor_name: bool = False,
        rewrite_llm: ChatClient | None = None,
        answer_cache: AnswerCache | None = None,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        if history_turns < 0:
            raise ValueError("history_turns must not be negative")
        if not -1.0 <= weak_cosine <= strong_cosine <= 1.0:
            raise ValueError("Evidence thresholds must satisfy -1 <= weak <= strong <= 1")
        self._retriever = retriever
        self._llm = llm
        self._rewrite_llm = rewrite_llm if rewrite_llm is not None else llm
        self._answerer = Answerer(llm, topics)
        self._candidates = candidates
        self._history_turns = min(history_turns, 2)
        self.topics = topics or FALLBACK_TOPICS
        self.faq_questions = faq_questions or FALLBACK_FAQ_QUESTIONS
        self._show_supervisor_name = show_supervisor_name
        self._answer_cache = answer_cache or AnswerCache()
        self._strong_cosine = strong_cosine
        self._weak_cosine = weak_cosine

    def answer(
        self,
        question: str,
        session: ChatSession,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        top_k: int = 3,
        use_reranker: bool = False,
    ) -> AnswerResult:
        """Answer one message, routing special intents before rewrite or retrieval."""
        display_question = question.strip()
        normalized_question = normalize_visitor_query(display_question)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        selected_mode = RetrievalMode(mode)
        started = perf_counter()
        input_check = check_input(display_question)
        flags = list(input_check.flags)
        intent = input_check.intent_override or input_check.intent

        if not display_question:
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
                display_question,
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
                display_question,
                session,
                selected_mode,
                "session_limit",
                Intent.OUT_OF_SCOPE,
                [*flags, "session_limit"],
                started,
                retain=False,
            )
        if intent in _BYPASS_INTENTS:
            status_reason = session.last_reason
            status_retry = session.last_retry_after_seconds
            status_text = system_status_text(status_reason, status_retry)
            return self._template_result(
                display_question,
                session,
                selected_mode,
                response_category(intent),
                intent,
                flags,
                started,
                reason=status_reason if intent is Intent.STATUS else AnswerReason.GUARD_TEMPLATE,
                retry_after_seconds=status_retry if intent is Intent.STATUS else None,
                status_text=status_text if intent is Intent.STATUS else None,
                faq_questions=self.faq_questions,
                update_last_reason=False,
            )

        history = session.history()
        context_history = _bounded_history(
            history[-self._history_turns :] if self._history_turns else []
        )
        rewrite_needed = should_rewrite(display_question, context_history)
        if not rewrite_needed:
            cached = self._answer_cache.get(
                normalize_query(normalized_question),
                selected_mode,
                top_k,
                PERSONA_VERSION,
            )
            if cached is not None:
                cached = replace(
                    cached,
                    standalone_query=normalize_query(normalized_question),
                    latency_ms={
                        "query_rewrite": 0.0,
                        "retrieval": 0.0,
                        "generation": 0.0,
                        "total": 0.0,
                    },
                    guardrail_flags=tuple(
                        dict.fromkeys((*cached.guardrail_flags, "answer_cache_hit"))
                    ),
                )
                session.record_exchange(
                    display_question,
                    cached.answer,
                    reason=cached.reason,
                    retry_after_seconds=cached.retry_after_seconds,
                )
                return cached

        standalone_query = normalize_query(normalized_question)
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
        hits, third_party_names = redact_retrieval_hits(
            hits, show_names=self._show_supervisor_name
        )
        if third_party_names:
            context_history = [
                replace(turn, content=redact_names(turn.content, third_party_names)[0])
                for turn in context_history
            ]
        rewrite_ms = 0.0

        if evidence_level != "none" and self._rewrite_llm is not None and rewrite_needed:
            rewrite_started = perf_counter()
            rewritten_query = rewrite(context_history, normalized_question, self._rewrite_llm)
            rewrite_ms = (perf_counter() - rewrite_started) * 1000
            rewritten_query = normalize_query(rewritten_query)
            if rewritten_query != standalone_query:
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
            answer = render_response(
                category,
                display_question,
                self.topics,
                previous_answer=_last_assistant_answer(session),
                faq_questions=self.faq_questions,
            )
            session.record_exchange(
                display_question,
                answer,
                retain_in_history=intent is Intent.NOT_IN_NOTES,
                reason=AnswerReason.GUARD_TEMPLATE,
            )
            total_ms = (perf_counter() - started) * 1000
            logger.info("Message routed: intent=%s flags=%s", intent.value, tuple(flags))
            logger.info("answer_reason=%s count=1", AnswerReason.GUARD_TEMPLATE.value)
            result = AnswerResult(
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
                reason=AnswerReason.GUARD_TEMPLATE,
            )
            return result

        intent = Intent.ABOUT_ME
        generation_started = perf_counter()
        generated = self._answerer.answer(
            standalone_query,
            display_question,
            hits,
            context_history,
            evidence_level,
            third_party_names,
        )
        generation_ms = (perf_counter() - generation_started) * 1000
        session.record_exchange(
            display_question,
            generated.answer,
            reason=generated.reason,
            retry_after_seconds=generated.retry_after_seconds,
        )
        all_flags = tuple(dict.fromkeys([*flags, *generated.guardrail_flags]))
        total_ms = (perf_counter() - started) * 1000
        logger.info("Message routed: intent=%s flags=%s", intent.value, all_flags)

        result = AnswerResult(
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
            reason=generated.reason,
            retry_after_seconds=generated.retry_after_seconds,
        )
        if not rewrite_needed and generated.reason is AnswerReason.OK and generated.used_llm:
            self._answer_cache.put(
                standalone_query,
                selected_mode,
                top_k,
                PERSONA_VERSION,
                result,
            )
        logger.info("answer_reason=%s count=1", generated.reason.value)
        return result

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
        reason: AnswerReason = AnswerReason.GUARD_TEMPLATE,
        retry_after_seconds: float | None = None,
        status_text: str | None = None,
        faq_questions: tuple[str, ...] = FALLBACK_FAQ_QUESTIONS,
        update_last_reason: bool = False,
    ) -> AnswerResult:
        answer = render_response(
            category,
            question,
            self.topics,
            previous_answer=_last_assistant_answer(session),
            status_text=status_text
            or "I don't have a more specific status to share right now.",
            retry_after_seconds=retry_after_seconds,
            faq_questions=faq_questions,
        )
        if question:
            session.record_exchange(
                question,
                answer,
                retain_in_history=retain,
                reason=reason,
                retry_after_seconds=retry_after_seconds,
                update_last_reason=update_last_reason,
            )
        all_flags = tuple(dict.fromkeys(flags))
        logger.info("Message routed: intent=%s flags=%s", intent.value, all_flags)
        logger.info("answer_reason=%s count=1", reason.value)
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
            reason=reason,
            retry_after_seconds=retry_after_seconds,
        )


def _last_assistant_answer(session: ChatSession) -> str | None:
    return next(
        (turn.content for turn in reversed(session.history()) if turn.role == "assistant"),
        None,
    )


def _bounded_history(history: list[Turn]) -> list[Turn]:
    """Keep the two most recent turns within the shared 1500-character cap."""
    selected = history[-2:]
    excess = sum(len(turn.content) for turn in selected) - 1_500
    bounded = list(selected)
    for index, turn in enumerate(bounded):
        if excess <= 0:
            break
        trimmed = min(excess, len(turn.content))
        bounded[index] = replace(turn, content=turn.content[trimmed:])
        excess -= trimmed
    return bounded
