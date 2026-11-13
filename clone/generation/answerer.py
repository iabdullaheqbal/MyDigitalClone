"""Grounded answer generation with output checks and friendly fallbacks."""

from __future__ import annotations

import logging
import re
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
from clone.guardrails.output_checks import check_output, redact_pii, repair_grounded_answer
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
                question,
                "Answer generation is not configured",
                AnswerReason.NO_KEY,
                hits=hits,
                evidence_level=evidence_level,
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
                hits=hits,
                evidence_level=evidence_level,
                retry_after_seconds=error.retry_after_seconds,
            )
        except GroqTimeoutError:
            return self._fallback(
                question,
                "The answer service timed out",
                AnswerReason.TIMEOUT,
                hits=hits,
                evidence_level=evidence_level,
            )
        except GroqEmptyOutputError:
            return self._fallback(
                question,
                "The answer service returned an empty reply",
                AnswerReason.EMPTY_LLM_OUTPUT,
                hits=hits,
                evidence_level=evidence_level,
            )
        except GroqRequestError:
            return self._fallback(
                question,
                "The answer service is temporarily unavailable",
                AnswerReason.LLM_HTTP_ERROR,
                hits=hits,
                evidence_level=evidence_level,
            )
        except GroqError:
            return self._fallback(
                question,
                "The answer service is temporarily unavailable",
                AnswerReason.LLM_HTTP_ERROR,
                hits=hits,
                evidence_level=evidence_level,
            )

        if not answer:
            return self._fallback(
                question,
                "The answer service returned an empty reply",
                AnswerReason.EMPTY_LLM_OUTPUT,
                hits=hits,
                evidence_level=evidence_level,
            )

        persona_facts = f"AI clone {PERSONA_NAME} {' '.join(PROJECTS)}"
        checked = check_output(
            answer,
            notes,
            persona_facts=persona_facts,
            topics=self.topics,
            third_party_names=third_party_names,
            user_message=question,
        )
        if checked.acceptable:
            return GenerationResult(checked.text, True, None)

        logger.info("groundedness_check_failed flags=%s", checked.flags)
        repaired = repair_grounded_answer(
            answer,
            notes,
            persona_facts=persona_facts,
            topics=self.topics,
            third_party_names=third_party_names,
            user_message=question,
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
        hits: list[RetrievalHit] | None = None,
        evidence_level: EvidenceLevel = "none",
    ) -> GenerationResult:
        extractive_reasons = {
            AnswerReason.NO_KEY,
            AnswerReason.RATE_LIMITED,
            AnswerReason.EMPTY_LLM_OUTPUT,
            AnswerReason.LLM_HTTP_ERROR,
            AnswerReason.TIMEOUT,
        }
        if evidence_level == "strong" and hits and reason_code in extractive_reasons:
            answer = _extractive_fallback(
                question,
                hits,
                reason_code,
                retry_after_seconds=retry_after_seconds,
            )
        elif aggregate_claim:
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


_EXTRACTIVE_STOP_WORDS = {
    "about",
    "and",
    "are",
    "can",
    "do",
    "for",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "the",
    "to",
    "what",
    "you",
    "your",
}


def _extractive_fallback(
    question: str,
    hits: list[RetrievalHit],
    reason: AnswerReason,
    *,
    retry_after_seconds: float | None = None,
) -> str:
    """Return up to two grounded, sentence-bounded excerpts from strong hits."""
    prefixes = {
        AnswerReason.NO_KEY: "My language model isn't connected, but here's what I can share:",
        AnswerReason.EMPTY_LLM_OUTPUT: (
            "My language model returned an empty reply, but here's what I can share:"
        ),
        AnswerReason.LLM_HTTP_ERROR: (
            "My language model had a temporary problem, but here's what I can share:"
        ),
        AnswerReason.TIMEOUT: (
            "My language model timed out, but here's what I can share:"
        ),
    }
    if reason is AnswerReason.RATE_LIMITED:
        seconds = max(1, round(retry_after_seconds or 25))
        prefix = (
            "My language model is busy right now; "
            f"please try again in about {seconds} seconds. "
            "Meanwhile, here's what my notes say:"
        )
    else:
        prefix = prefixes[reason]
    query_terms = set(re.findall(r"[a-z0-9]+", question.casefold())) - _EXTRACTIVE_STOP_WORDS
    candidates: list[tuple[int, int, int, str, str]] = []
    for hit_index, hit in enumerate(hits):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", hit.chunk.text.strip())
        for sentence_index, sentence in enumerate(sentences):
            clean_sentence = sentence.strip()
            word_count = len(clean_sentence.split())
            if not clean_sentence or word_count > 120:
                continue
            sentence_terms = set(re.findall(r"[a-z0-9]+", clean_sentence.casefold()))
            candidates.append(
                (
                    len(query_terms & sentence_terms),
                    -hit_index,
                    -sentence_index,
                    hit.chunk.source,
                    clean_sentence,
                )
            )
    selected: list[tuple[str, str]] = []
    words_used = 0
    for _, _, _, source, sentence in sorted(candidates, reverse=True):
        sentence_words = len(sentence.split())
        if words_used + sentence_words > 120:
            continue
        selected.append((source, redact_pii(sentence)))
        words_used += sentence_words
        if len(selected) == 2:
            break
    if not selected:
        return "I couldn't extract a short passage safely, so I won't guess."
    excerpts = "\n\n".join(f"From {source}: {sentence}" for source, sentence in selected)
    return f"{prefix}\n\n{excerpts}"
