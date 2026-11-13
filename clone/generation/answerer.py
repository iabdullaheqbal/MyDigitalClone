"""Grounded answer generation with output checks and friendly fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from clone.generation.guard import EvidenceLevel
from clone.generation.llm_client import ChatClient, GroqError, GroqRateLimitError
from clone.generation.prompts import answer_messages
from clone.generation.responses import render_response
from clone.generation.topics import FALLBACK_TOPICS, project_examples
from clone.guardrails.claims import aggregate_subject, is_aggregate_claim
from clone.guardrails.output_checks import check_output
from clone.guardrails.privacy import redact_names
from clone.persona import PERSONA_NAME
from clone.types import RetrievalHit, Turn

logger = logging.getLogger(__name__)
MAX_COMPLETION_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Answer text and safe generation diagnostics."""

    answer: str
    used_llm: bool
    error: str | None
    guardrail_flags: tuple[str, ...] = ()


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
                question,
                "Answer generation is not configured",
                aggregate_claim=aggregate_claim,
                hits=hits,
            )

        notes = "\n\n".join(hit.chunk.text for hit in hits)
        flags: list[str] = []
        for attempt in range(2):
            messages = answer_messages(
                standalone_query,
                question,
                hits,
                safe_history[-6:],
                topics=self.topics,
                evidence_level=evidence_level,
                strict=attempt == 1,
                aggregate_claim=aggregate_claim,
            )
            try:
                answer = self.llm.chat(
                    messages,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0.2,
                ).strip()
            except GroqError as error:
                logger.warning("Answer generation failed: %s", type(error).__name__)
                reason = (
                    "The answer service is temporarily rate-limited"
                    if isinstance(error, GroqRateLimitError)
                    else "The answer service is temporarily unavailable"
                )
                return self._fallback(
                    question,
                    reason,
                    tuple(flags),
                    aggregate_claim=aggregate_claim,
                    hits=hits,
                )

            checked = check_output(
                answer,
                notes,
                persona_facts=f"AI clone {PERSONA_NAME}",
                topics=self.topics,
                third_party_names=third_party_names,
            )
            if checked.acceptable:
                return GenerationResult(checked.text, True, None, tuple(flags))
            flags.extend(checked.flags)

        logger.info("Rejected model answer: flags=%s", tuple(dict.fromkeys(flags)))
        return self._fallback(
            question,
            "I couldn't verify a safe answer from the notes",
            tuple(dict.fromkeys(flags)),
            aggregate_claim=aggregate_claim,
            hits=hits,
        )

    def _fallback(
        self,
        question: str,
        reason: str,
        flags: tuple[str, ...] = (),
        *,
        aggregate_claim: bool = False,
        hits: list[RetrievalHit] | None = None,
    ) -> GenerationResult:
        if aggregate_claim:
            answer = render_response(
                "aggregate_hedge",
                question,
                self.topics,
                examples=project_examples(hit.chunk.source for hit in hits or []),
                subject=aggregate_subject(question),
            )
        else:
            answer = render_response("unavailable", question, self.topics)
        return GenerationResult(
            answer,
            False,
            reason,
            flags,
        )
