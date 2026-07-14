"""Embedding model wrapper using sentence-transformers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Lazy-loaded sentence-transformer embedding model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        return int(model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        # BGE models benefit from a retrieval instruction for queries only;
        # documents are embedded as-is.
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        # BGE retrieval convention: prefix queries for better recall.
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vector = model.encode(
            [prefixed],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector[0], dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
