"""Format service results for the Gradio interface."""

from __future__ import annotations

from clone.guardrails.output_checks import redact_pii
from clone.types import AnswerResult


def format_evidence(result: AnswerResult) -> dict[str, object]:
    """Return display-safe retrieval details and per-stage timings."""
    return {
        "intent": result.intent,
        "evidence_level": result.evidence_level,
        "guardrail_flags": list(result.guardrail_flags),
        "not_found_in_notes_badge": ("Not found in notes" if result.not_found_in_notes else ""),
        "retrieval_query": result.standalone_query,
        "retrieval_mode": result.mode.value,
        "declined": result.declined,
        "used_groq": result.used_llm,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "passages": [
            {
                "source": hit.chunk.source,
                "chunk_id": hit.chunk.id,
                "text": redact_pii(hit.chunk.text),
                "score": hit.score,
                "bm25_score": hit.bm25_score,
                "dense_score": hit.dense_score,
                "bm25_rank": hit.bm25_rank,
                "dense_rank": hit.dense_rank,
                "rrf_score": hit.rrf_score,
                "rerank_score": hit.rerank_score,
            }
            for hit in result.hits
        ],
    }
