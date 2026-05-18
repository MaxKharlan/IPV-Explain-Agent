"""Embedding-based retrieval for the local methodology corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.rag.indexer import RAGChunk, build_chunk_index


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingBackend(Protocol):
    """Protocol for embedding providers used by the retriever."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embeds a batch of texts into a 2D float array."""


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk with its semantic similarity score."""

    chunk: RAGChunk
    score: float


class SentenceTransformerEmbeddingBackend:
    """Embedding backend powered by sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install project dependencies to enable embedding-based retrieval."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)

        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalizes vectors row-wise for cosine similarity."""
    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class InMemoryVectorRetriever:
    """Semantic retriever over an in-memory chunk index."""

    def __init__(self, chunks: list[RAGChunk], embedding_backend: EmbeddingBackend) -> None:
        self.chunks = chunks
        self.embedding_backend = embedding_backend
        self._chunk_embeddings = self._build_chunk_embeddings()

    def _build_chunk_embeddings(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0, 0), dtype=float)

        texts = [chunk.text for chunk in self.chunks]
        vectors = self.embedding_backend.embed_texts(texts)
        return _normalize_vectors(np.asarray(vectors, dtype=float))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Returns top-k semantically closest chunks for the query."""
        if not query.strip() or not self.chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.embedding_backend.embed_texts([query])
        normalized_query = _normalize_vectors(np.asarray(query_vector, dtype=float))
        if normalized_query.size == 0:
            return []

        scores = self._chunk_embeddings @ normalized_query[0]
        ranked_indices = np.argsort(scores)[::-1]

        results: list[RetrievalResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            results.append(RetrievalResult(chunk=self.chunks[index], score=score))
            if len(results) == top_k:
                break
        return results


def build_semantic_retriever(
    corpus_root: str | Path,
    *,
    embedding_backend: EmbeddingBackend | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> InMemoryVectorRetriever:
    """Builds a semantic retriever for the methodology corpus."""
    backend = embedding_backend or SentenceTransformerEmbeddingBackend()
    chunks = build_chunk_index(
        corpus_root,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return InMemoryVectorRetriever(chunks=chunks, embedding_backend=backend)
