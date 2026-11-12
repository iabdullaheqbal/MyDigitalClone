"""Build committed chunk and embedding artifacts from the approved notes."""

from __future__ import annotations

from clone.config import Settings
from clone.indexing.builder import build_index


def main() -> None:
    """Load the configured CPU encoder and build the local index."""
    from sentence_transformers import SentenceTransformer

    settings = Settings.from_env()
    encoder = SentenceTransformer(settings.embed_model, device="cpu")
    chunks, _ = build_index(
        data_dir=settings.data_dir,
        encoder=encoder,
        out_dir=settings.artifacts_dir,
        model=settings.embed_model,
        chunk_words=settings.chunk_words,
        overlap_words=settings.overlap_words,
    )
    print(f"Built index with {len(chunks)} chunks using {settings.embed_model}.")


if __name__ == "__main__":
    main()
