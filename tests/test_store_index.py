from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from clone.indexing.builder import build_index
from clone.indexing.store import fingerprint, load_index, load_topics


class FakeEncoder:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> NDArray[np.float32]:
        assert normalize_embeddings
        vectors = np.zeros((len(sentences), 8), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for word in sentence.lower().split():
                vectors[row, sum(word.encode("utf-8")) % vectors.shape[1]] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1)


def test_build_and_load_round_trip_with_fingerprint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text(
        "# Bird watching\nFirst paragraph.\n\n## Moon formations\nSecond paragraph.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "artifacts"

    chunks, embeddings = build_index(
        data_dir,
        FakeEncoder(),
        out_dir,
        model="fake-model",
        chunk_words=4,
        overlap_words=1,
    )
    expected = fingerprint(data_dir, "fake-model", 4, 1)
    loaded_chunks, loaded_embeddings = load_index(out_dir, expected)

    assert loaded_chunks == chunks
    np.testing.assert_allclose(loaded_embeddings, embeddings)
    assert loaded_embeddings.dtype == np.float32
    assert load_topics(out_dir) == ("Bird watching", "Moon formations", "notes")


def test_fingerprint_changes_with_data_model_and_chunk_configuration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "notes.txt"
    source.write_text("original", encoding="utf-8")
    original = fingerprint(data_dir, "model-a", 250, 40)

    assert fingerprint(data_dir, "model-b", 250, 40) != original
    assert fingerprint(data_dir, "model-a", 200, 40) != original
    source.write_text("changed", encoding="utf-8")
    assert fingerprint(data_dir, "model-a", 250, 40) != original


def test_load_rejects_stale_and_shape_mismatched_index(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.txt").write_text("one two three", encoding="utf-8")
    out_dir = tmp_path / "artifacts"
    build_index(data_dir, FakeEncoder(), out_dir, chunk_words=10, overlap_words=2)

    with pytest.raises(ValueError, match="fingerprint"):
        load_index(out_dir, "stale")

    np.save(out_dir / "embeddings.npy", np.ones((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="incompatible"):
        load_index(out_dir)
