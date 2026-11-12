"""Load owner-provided Markdown and text documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A named UTF-8 document."""

    name: str
    text: str


def load_documents(data_dir: Path) -> list[Document]:
    """Read supported documents in stable name order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory does not exist: {data_dir}")

    paths = sorted(
        (
            path
            for path in data_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("_")
            and path.suffix.lower() in {".md", ".txt"}
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not paths:
        raise ValueError(f"No .md or .txt documents found in {data_dir}")

    documents = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"Document is not valid UTF-8: {path}") from error
        documents.append(Document(name=path.name, text=text))
    return documents
