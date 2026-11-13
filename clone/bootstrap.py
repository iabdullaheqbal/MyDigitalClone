"""Construct the application service and Gradio interface from configuration."""

from __future__ import annotations

import gradio as gr

from clone.config import Settings
from clone.generation.llm_client import GroqClient
from clone.indexing.store import fingerprint, load_index, load_topics
from clone.retrieval.bm25 import BM25Retriever
from clone.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from clone.retrieval.hybrid import HybridRetriever
from clone.retrieval.reranker import CrossEncoderReranker
from clone.service import CloneService
from clone.ui.layout import build_ui


def create_app() -> gr.Blocks:
    """Load the prebuilt index and assemble the app without starting its server."""
    settings = Settings.from_env()
    index_fingerprint = fingerprint(
        settings.data_dir,
        settings.embed_model,
        settings.chunk_words,
        settings.overlap_words,
    )
    chunks, embeddings = load_index(settings.artifacts_dir, index_fingerprint)
    encoder = SentenceTransformerEncoder(settings.embed_model)
    retriever = HybridRetriever(
        BM25Retriever(chunks),
        DenseRetriever(chunks, embeddings, encoder),
        rrf_k=settings.rrf_k,
        reranker=CrossEncoderReranker(),
    )
    llm = (
        GroqClient(settings.groq_api_key, settings.groq_model, settings.groq_base_url)
        if settings.groq_api_key
        else None
    )
    service = CloneService(
        retriever,
        llm,
        strong_cosine=settings.strong_cosine,
        weak_cosine=settings.weak_cosine,
        candidates=settings.candidates,
        history_turns=settings.history_turns,
        topics=load_topics(settings.artifacts_dir),
        show_supervisor_name=settings.show_supervisor_name,
    )
    return build_ui(service, reranker_available=True)
