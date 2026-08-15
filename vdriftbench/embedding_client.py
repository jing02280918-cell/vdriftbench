"""Embedding backend for Layer 2 semantic drift and the Layer 1 geometry
fast-path (method sections 四 and 七 — both consume the same vectors).

`SentenceTransformerEmbeddingClient` wraps an open-source multilingual model
(BGE-M3 by default, per the earlier discussion on open-source Chinese-capable
embedding models). `MockEmbeddingClient` produces a deterministic
hash-derived unit vector so the pipeline runs without downloading any model.
"""

from __future__ import annotations

import abc
import hashlib
import numpy as np


class EmbeddingClient(abc.ABC):
    @abc.abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalized embedding vector for `text`."""


class SentenceTransformerEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEmbeddingClient requires: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


class MockEmbeddingClient(EmbeddingClient):
    """Deterministic offline stand-in — same text always maps to the same
    vector, and unrelated texts are (with high probability) near-orthogonal,
    which is enough to exercise the cosine-similarity plumbing without a
    real model."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_embedding_client(model_name: str, mock: bool, device: str = "cuda") -> EmbeddingClient:
    if mock:
        return MockEmbeddingClient()
    return SentenceTransformerEmbeddingClient(model_name=model_name, device=device)
