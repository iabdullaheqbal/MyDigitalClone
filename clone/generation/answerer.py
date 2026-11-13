"""Grounded answer generation with output checks and friendly fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clone.generation.guard import EvidenceLevel
from clone.generation.llm_client import ChatClient, GroqError, GroqRateLimitError
from clone.generation.prompts import answer_messages
from clone.generation.responses import render_response
from clone.generation.topics import FALLBACK_TOPICS
from clone.guardrails.output_checks import check_output
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
    ) -> GenerationResult:
        """Generate, validate, and return a safe answer from graded evidence."""
        if self.llm is None:
            return self._fallback(question, "Answer generation is not configured")

        notes = "\n\n".join(hit.chunk.text for hit in hits)
        flags: list[str] = []
        for attempt in range(2):
            messages = answer_messages(
                standalone_query,
                question,
                hits,
                history[-6:],
                topics=self.topics,
                evidence_level=evidence_level,
                strict=attempt == 1,
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
                return self._fallback(question, reason, tuple(flags))

            checked = check_output(
                answer,
                notes,
                persona_facts=f"AI clone {PERSONA_NAME}",
                topics=self.topics,
            )
            if checked.acceptable:
                return GenerationResult(checked.text, True, None, tuple(flags))
            flags.extend(checked.flags)

        logger.info("Rejected model answer: flags=%s", tuple(dict.fromkeys(flags)))
        return self._fallback(
            question,
            "I couldn't verify a safe answer from the notes",
            tuple(dict.fromkeys(flags)),
        )

    def _fallback(
        self,
        question: str,
        reason: str,
        flags: tuple[str, ...] = (),
    ) -> GenerationResult:
        return GenerationResult(
            render_response("unavailable", question, self.topics),
            False,
            reason,
            flags,
        )
