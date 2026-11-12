from pathlib import Path

import pytest

from clone.ingestion.loader import load_documents


def test_loader_sorts_supported_utf8_documents_and_skips_underscores(tmp_path: Path) -> None:
    (tmp_path / "zeta.TXT").write_text("café", encoding="utf-8")
    (tmp_path / "Alpha.md").write_text("# naïve", encoding="utf-8")
    (tmp_path / "_private.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.name for document in documents] == ["Alpha.md", "zeta.TXT"]
    assert documents[1].text == "café"


def test_loader_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .md or .txt"):
        load_documents(tmp_path)


def test_loader_reports_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "broken.md").write_bytes(b"\xff")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_documents(tmp_path)
