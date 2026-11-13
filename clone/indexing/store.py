"""Persist and validate chunk metadata and dense embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clone.ingestion.loader import load_documents
from clone.types import Chunk


def fingerprint(
    data_dir: Path,
    model: str,
    chunk_words: int,
    overlap_words: int,
) -> str:
    """Hash the source documents and settings that determine the index."""
    documents = load_documents(data_dir)
    manifest: dict[str, Any] = {
        "model": model,
        "chunk_config": {"chunk_words": chunk_words, "overlap_words": overlap_words},
        "documents": [
            {
                "name": document.name,
                "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            }
            for document in documents
        ],
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_embeddings(
    embeddings: NDArray[np.float32],
    chunk_count: int,
    normalize: bool,
) -> NDArray[np.float32]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != chunk_count or matrix.shape[1] == 0:
        raise ValueError(
            f"Embedding shape {matrix.shape} is incompatible with {chunk_count} chunks"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values")

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings contain zero-length vectors")
    if normalize:
        matrix = matrix / norms[:, np.newaxis]
    elif not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Index embeddings are not normalized")
    return np.asarray(matrix, dtype=np.float32)


def save_index(
    out_dir: Path,
    chunks: list[Chunk],
    embeddings: NDArray[np.float32],
    model: str,
    chunk_words: int,
    overlap_words: int,
    data_fingerprint: str,
    topics: list[str] | None = None,
) -> None:
    """Write a normalized NumPy matrix and its matching JSON metadata."""
    if not chunks:
        raise ValueError("Cannot save an index without chunks")
    matrix = _validate_embeddings(embeddings, len(chunks), normalize=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "fingerprint": data_fingerprint,
        "model": model,
        "chunk_config": {"chunk_words": chunk_words, "overlap_words": overlap_words},
        "topics": topics or [],
        "chunks": [
            {"id": chunk.id, "source": chunk.source, "text": chunk.text, "index": chunk.index}
            for chunk in chunks
        ],
    }
    with tempfile.TemporaryDirectory(dir=out_dir) as temporary:
        temporary_dir = Path(temporary)
        np.save(temporary_dir / "embeddings.npy", matrix, allow_pickle=False)
        (temporary_dir / "chunks.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir / "embeddings.npy", out_dir / "embeddings.npy")
        os.replace(temporary_dir / "chunks.json", out_dir / "chunks.json")


def load_index(
    out_dir: Path,
    expected_fingerprint: str | None = None,
) -> tuple[list[Chunk], NDArray[np.float32]]:
    """Load an index, rejecting incomplete, stale, or mismatched artifacts."""
    metadata_path = out_dir / "chunks.json"
    embeddings_path = out_dir / "embeddings.npy"
    missing = [path.name for path in (metadata_path, embeddings_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Index is incomplete; missing: {', '.join(missing)}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_fingerprint = metadata.get("fingerprint")
    if not isinstance(saved_fingerprint, str) or not saved_fingerprint:
        raise ValueError("Index metadata is missing its fingerprint")
    if expected_fingerprint is not None and saved_fingerprint != expected_fingerprint:
        raise ValueError("Index fingerprint does not match the current data and settings")

    raw_chunks = metadata.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("Index metadata contains no chunks")
    try:
        chunks = [Chunk(**item) for item in raw_chunks]
    except (TypeError, KeyError) as error:
        raise ValueError("Index metadata contains an invalid chunk entry") from error

    embeddings = np.load(embeddings_path, allow_pickle=False)
    matrix = _validate_embeddings(embeddings, len(chunks), normalize=False)
    return chunks, matrix


def load_topics(out_dir: Path) -> tuple[str, ...]:
    """Load curated topics from indexed source filenames."""
    from clone.generation.topics import FALLBACK_TOPICS, topics_for_sources

    metadata_path = out_dir / "chunks.json"
    if not metadata_path.is_file():
        return FALLBACK_TOPICS
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_chunks = metadata.get("chunks")
    if isinstance(raw_chunks, list):
        sources = [
            chunk["source"]
            for chunk in raw_chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("source"), str)
        ]
        if sources:
            return tuple(topics_for_sources(sources))
    return FALLBACK_TOPICS
