"""Grounded answer generation with output checks and friendly fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from clone.generation.guard import EvidenceLevel
from clone.generation.llm_client import (
    ChatClient,
    GroqEmptyOutputError,
    GroqError,
    GroqRateLimitError,
    GroqRequestError,
    GroqTimeoutError,
)
from clone.generation.prompts import answer_messages
from clone.generation.responses import render_response, system_status_text
from clone.generation.topics import FALLBACK_TOPICS, PROJECTS
from clone.guardrails.claims import aggregate_subject, is_aggregate_claim
from clone.guardrails.output_checks import check_output, repair_grounded_answer
from clone.guardrails.privacy import redact_names
from clone.persona import PERSONA_NAME
from clone.types import AnswerReason, RetrievalHit, Turn

logger = logging.getLogger(__name__)
MAX_COMPLETION_TOKENS = 700


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Answer text and safe generation diagnostics."""

    answer: str
    used_llm: bool
    error: str | None
    guardrail_flags: tuple[str, ...] = ()
    reason: AnswerReason = AnswerReason.OK
    retry_after_seconds: float | None = None


class Answerer:
    """Generate an answer from current-turn evidence only."""

    def __init__(
        self,
        llm: ChatClient | None,
        topics: tuple[str, ...] = FALLBACK_TOPICS,
    ) -> None:
        self.llm = llm
        self.topics = topics or FALLBACK_TOPICS

    def answer(
        self,
        standalone_query: str,
        question: str,
        hits: list[RetrievalHit],
        history: list[Turn],
        evidence_level: EvidenceLevel,
        third_party_names: tuple[str, ...] = (),
    ) -> GenerationResult:
        """Generate, validate, and return a safe answer from graded evidence."""
        aggregate_claim = is_aggregate_claim(question)
        question, _ = redact_names(question, third_party_names)
        standalone_query, _ = redact_names(standalone_query, third_party_names)
        safe_history = [
            replace(turn, content=redact_names(turn.content, third_party_names)[0])
            for turn in history
        ]
        if self.llm is None:
            return self._fallback(
                question, "Answer generation is not configured", AnswerReason.NO_KEY
            )

        notes = "\n\n".join(hit.chunk.text for hit in hits)
        messages = answer_messages(
            standalone_query,
            question,
            hits,
            safe_history[-2:],
            topics=self.topics,
            evidence_level=evidence_level,
            aggregate_claim=aggregate_claim,
        )
        try:
            answer = self.llm.chat(
                messages,
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.2,
            ).strip()
        except GroqRateLimitError as error:
            return self._fallback(
                question,
                "The answer service is temporarily rate-limited",
                AnswerReason.RATE_LIMITED,
                retry_after_seconds=error.retry_after_seconds,
            )
        except GroqTimeoutError:
            return self._fallback(
                question,
                "The answer service timed out",
                AnswerReason.TIMEOUT,
            )
        except GroqEmptyOutputError:
            return self._fallback(
                question,
                "The answer service returned an empty reply",
                AnswerReason.EMPTY_LLM_OUTPUT,
            )
        except GroqRequestError:
            return self._fallback(
                question,
                "The answer service is temporarily unavailable",
                AnswerReason.LLM_HTTP_ERROR,
            )
        except GroqError:
            return self._fallback(
                question,
                "The answer service is temporarily unavailable",
                AnswerReason.LLM_HTTP_ERROR,
            )

        checked = check_output(
            answer,
            notes,
            persona_facts=f"AI clone {PERSONA_NAME} {' '.join(PROJECTS)}",
            topics=self.topics,
            third_party_names=third_party_names,
        )
        if checked.acceptable:
            return GenerationResult(checked.text, True, None)

        repaired = repair_grounded_answer(
            answer,
            notes,
            persona_facts=f"AI clone {PERSONA_NAME} {' '.join(PROJECTS)}",
            topics=self.topics,
            third_party_names=third_party_names,
        )
        if repaired.acceptable:
            logger.info("Answer repaired: reason=%s count=1", AnswerReason.GROUNDEDNESS_FAILED)
            return GenerationResult(
                repaired.text,
                True,
                None,
                repaired.flags,
                AnswerReason.GROUNDEDNESS_FAILED,
            )
        return self._fallback(
            question,
            "I couldn't verify a safe answer from the notes",
            AnswerReason.GROUNDEDNESS_FAILED,
            repaired.flags or checked.flags,
            aggregate_claim=aggregate_claim,
        )

    def _fallback(
        self,
        question: str,
        reason: str,
        reason_code: AnswerReason,
        flags: tuple[str, ...] = (),
        *,
        aggregate_claim: bool = False,
        retry_after_seconds: float | None = None,
    ) -> GenerationResult:
        if aggregate_claim:
            answer = render_response(
                "aggregate_hedge",
                question,
                self.topics,
                subject=aggregate_subject(question),
            )
        elif reason_code is AnswerReason.RATE_LIMITED:
            answer = render_response(
                "rate_limited",
                question,
                self.topics,
                retry_after_seconds=retry_after_seconds,
            )
        else:
            answer = render_response(
                "unavailable",
                question,
                self.topics,
                status_text=system_status_text(reason_code, retry_after_seconds),
            )
        return GenerationResult(
            answer,
            False,
            reason,
            flags,
            reason_code,
            retry_after_seconds,
        )
