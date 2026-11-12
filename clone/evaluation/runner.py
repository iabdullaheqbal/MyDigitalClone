"""Run offline source-relevance evaluations over retrieval modes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Protocol

from clone.config import Settings
from clone.evaluation.dataset import EvaluationDataset, MultiTurnDataset
from clone.evaluation.metrics import RetrievalMetrics, score_retrieval
from clone.indexing.store import fingerprint, load_index
from clone.retrieval.bm25 import BM25Retriever
from clone.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from clone.retrieval.hybrid import HybridRetriever
from clone.types import RetrievalHit, RetrievalMode, Turn


class Retriever(Protocol):
    """Retrieval interface used by offline evaluation."""

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        candidates: int = 20,
        use_reranker: bool = False,
    ) -> list[RetrievalHit]: ...


QueryRewriter = Callable[[list[Turn], str], str]


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Metrics and timings for one evaluated query."""

    case_id: str
    question: str
    standalone_query: str
    expected_standalone_query: str | None
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    rewrite_latency_ms: float
    retrieval_latency_ms: float
    metrics: RetrievalMetrics


@dataclass(frozen=True, slots=True)
class ModeEvaluation:
    """Per-mode aggregate metrics and individual query results."""

    mode: str
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank_at_k: float
    p50_retrieval_latency_ms: float
    queries: tuple[QueryEvaluation, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Evaluation output, including draft status to prevent false claims."""

    dataset: str
    review_status: str
    human_review_required: bool
    k: int
    modes: tuple[ModeEvaluation, ...]


@dataclass(frozen=True, slots=True)
class _Query:
    case_id: str
    question: str
    relevant_sources: tuple[str, ...]
    history: tuple[Turn, ...] = ()
    expected_standalone_query: str | None = None


def build_retriever(settings: Settings) -> HybridRetriever:
    """Load the configured index and assemble retrieval without an LLM."""
    expected = fingerprint(
        settings.data_dir,
        settings.embed_model,
        settings.chunk_words,
        settings.overlap_words,
    )
    chunks, embeddings = load_index(settings.artifacts_dir, expected)
    encoder = SentenceTransformerEncoder(settings.embed_model)
    return HybridRetriever(
        BM25Retriever(chunks),
        DenseRetriever(chunks, embeddings, encoder),
        rrf_k=settings.rrf_k,
    )


def run_retrieval_evaluation(
    dataset: EvaluationDataset,
    retriever: Retriever,
    *,
    modes: Sequence[RetrievalMode] | None = None,
    k: int = 5,
    candidates: int = 20,
) -> EvaluationReport:
    """Measure source-level retrieval quality without calling a generation API."""
    queries = [
        _Query(case.id, case.question, case.relevant_sources)
        for case in dataset.cases
    ]
    return _run(
        dataset.name,
        dataset.review_status,
        dataset.human_review_required,
        queries,
        retriever,
        modes=modes,
        k=k,
        candidates=candidates,
    )


def run_multiturn_retrieval_evaluation(
    dataset: MultiTurnDataset,
    retriever: Retriever,
    query_rewriter: QueryRewriter,
    *,
    modes: Sequence[RetrievalMode] | None = None,
    k: int = 5,
    candidates: int = 20,
) -> EvaluationReport:
    """Measure retrieval after an injected follow-up query rewrite."""
    queries = [
        _Query(
            case.id,
            case.question,
            case.relevant_sources,
            case.history,
            case.expected_standalone_query,
        )
        for case in dataset.cases
    ]
    return _run(
        dataset.name,
        dataset.review_status,
        dataset.human_review_required,
        queries,
        retriever,
        query_rewriter=query_rewriter,
        modes=modes,
        k=k,
        candidates=candidates,
    )


def _run(
    dataset_name: str,
    review_status: str,
    human_review_required: bool,
    queries: list[_Query],
    retriever: Retriever,
    *,
    query_rewriter: QueryRewriter | None = None,
    modes: Sequence[RetrievalMode] | None = None,
    k: int = 5,
    candidates: int = 20,
) -> EvaluationReport:
    if not queries:
        raise ValueError("Evaluation requires at least one query")
    if k < 1 or candidates < 1:
        raise ValueError("k and candidates must be at least 1")
    selected_modes = tuple(
        RetrievalMode(mode) for mode in (tuple(RetrievalMode) if modes is None else modes)
    )
    if not selected_modes or len(set(selected_modes)) != len(selected_modes):
        raise ValueError("Evaluation modes must be non-empty and unique")

    prepared_queries = []
    for query in queries:
        if query.history and query_rewriter is not None:
            started = perf_counter()
            standalone = query_rewriter(list(query.history), query.question)
            rewrite_latency = (perf_counter() - started) * 1000
        else:
            standalone = query.question
            rewrite_latency = 0.0
        prepared_queries.append((query, standalone, rewrite_latency))

    mode_results = []
    for mode in selected_modes:
        query_results = []
        for query, standalone, rewrite_latency in prepared_queries:
            started = perf_counter()
            hits = retriever.retrieve(
                standalone,
                mode,
                k=k,
                candidates=candidates,
            )
            retrieval_latency = (perf_counter() - started) * 1000
            sources = tuple(hit.chunk.source for hit in hits)
            metrics = score_retrieval(sources, query.relevant_sources, k)
            query_results.append(
                QueryEvaluation(
                    case_id=query.case_id,
                    question=query.question,
                    standalone_query=standalone,
                    expected_standalone_query=query.expected_standalone_query,
                    expected_sources=query.relevant_sources,
                    retrieved_sources=sources,
                    rewrite_latency_ms=rewrite_latency,
                    retrieval_latency_ms=retrieval_latency,
                    metrics=metrics,
                )
            )
        mode_results.append(
            ModeEvaluation(
                mode=mode.value,
                mean_precision_at_k=_mean(
                    item.metrics.precision_at_k for item in query_results
                ),
                mean_recall_at_k=_mean(item.metrics.recall_at_k for item in query_results),
                mean_reciprocal_rank_at_k=_mean(
                    item.metrics.reciprocal_rank_at_k for item in query_results
                ),
                p50_retrieval_latency_ms=median(
                    item.retrieval_latency_ms for item in query_results
                ),
                queries=tuple(query_results),
            )
        )
    return EvaluationReport(
        dataset=dataset_name,
        review_status=review_status,
        human_review_required=human_review_required,
        k=k,
        modes=tuple(mode_results),
    )


def _mean(values: Iterable[float]) -> float:
    numbers = tuple(values)
    if not numbers:
        raise ValueError("Cannot calculate a mean without values")
    return sum(numbers) / len(numbers)
