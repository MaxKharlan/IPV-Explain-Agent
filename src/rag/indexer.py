"""Utilities for indexing the local methodology corpus for RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class RAGDocument:
    """One source document from the local methodology corpus."""

    doc_id: str
    path: str
    title: str
    text: str


@dataclass(frozen=True)
class RAGChunk:
    """One chunk of a source document prepared for retrieval."""

    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str


def discover_corpus_files(corpus_root: str | Path) -> list[Path]:
    """Recursively finds supported corpus files under the given root."""
    root = Path(corpus_root)
    if not root.exists():
        return []

    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
    ]
    return sorted(files)


def load_corpus_documents(corpus_root: str | Path) -> list[RAGDocument]:
    """Loads all supported text documents from the corpus root recursively."""
    root = Path(corpus_root)
    documents: list[RAGDocument] = []
    for path in discover_corpus_files(root):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        relative = path.relative_to(root).as_posix()
        documents.append(
            RAGDocument(
                doc_id=relative,
                path=str(path),
                title=path.stem.replace("_", " "),
                text=text,
            )
        )
    return documents


def normalize_text(text: str) -> str:
    """Normalizes whitespace to make chunking and retrieval more stable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Splits text into overlapping chunks.

    This logic does not rely on any specific corpus folder structure.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step
    return chunks


def build_chunk_index(
    corpus_root: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RAGChunk]:
    """Builds an in-memory chunk index for the whole corpus."""
    chunks: list[RAGChunk] = []
    for document in load_corpus_documents(corpus_root):
        for index, chunk_text in enumerate(
            split_text_into_chunks(
                document.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            chunks.append(
                RAGChunk(
                    chunk_id=f"{document.doc_id}::chunk-{index}",
                    doc_id=document.doc_id,
                    source_path=document.path,
                    title=document.title,
                    text=chunk_text,
                )
            )
    return chunks
