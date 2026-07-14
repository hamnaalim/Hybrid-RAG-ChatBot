"""Cross-encoder reranking."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from app.config import settings
from app.retriever import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank candidate chunks with a cross-encoder relevance model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k
        model = self._load()
        pairs = [(query, c.chunk.content) for c in candidates]
        scores = model.predict(pairs)

        reranked: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            merged_scores = {**candidate.scores, "rerank": float(score)}
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    scores=merged_scores,
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
