"""Format service results for the Gradio interface."""

from __future__ import annotations

from clone.types import AnswerResult


def format_evidence(result: AnswerResult) -> dict[str, object]:
    """Return display-safe retrieval details and per-stage timings."""
    return {
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
                "text": hit.chunk.text,
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
