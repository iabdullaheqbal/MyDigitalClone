"""Build persistent chunk and embedding artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from clone.generation.faq import extract_faq_questions
from clone.generation.topics import derive_topics
from clone.indexing.store import fingerprint, save_index
from clone.ingestion.chunker import chunk_text
from clone.ingestion.loader import load_documents
from clone.types import Chunk


class Encoder(Protocol):
    """Minimal interface required by the index builder."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> NDArray[np.float32]: ...


def build_index(
    data_dir: Path,
    encoder: Encoder,
    out_dir: Path,
    model: str = "all-MiniLM-L6-v2",
    chunk_words: int = 250,
    overlap_words: int = 40,
) -> tuple[list[Chunk], NDArray[np.float32]]:
    """Chunk documents, encode them once, and persist the resulting index."""
    documents = load_documents(data_dir)
    chunks = []
    for document in documents:
        for index, text in enumerate(chunk_text(document.text, chunk_words, overlap_words)):
            chunks.append(
                Chunk(
                    id=f"{Path(document.name).stem}#{index}",
                    source=document.name,
                    text=text,
                    index=len(chunks),
                )
            )
    if not chunks:
        raise ValueError(f"No non-empty chunks were produced from {data_dir}")

    embeddings = np.asarray(
        encoder.encode([chunk.text for chunk in chunks], normalize_embeddings=True),
        dtype=np.float32,
    )
    data_fingerprint = fingerprint(data_dir, model, chunk_words, overlap_words)
    save_index(
        out_dir,
        chunks,
        embeddings,
        model,
        chunk_words,
        overlap_words,
        data_fingerprint,
        derive_topics(documents),
        extract_faq_questions(documents),
    )
    return chunks, embeddings
