"""Citation verification via NLI, semantic similarity, or LLM-as-judge."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

import numpy as np

from app.config import settings
from app.embeddings import EmbeddingModel, get_embedding_model
from app.generator import Citation, GenerationResult

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verify that cited passages support the claims in the answer."""

    def __init__(
        self,
        mode: Literal["nli", "similarity", "llm"] | None = None,
        embedding_model: EmbeddingModel | None = None,
        nli_model_name: str | None = None,
        similarity_threshold: float | None = None,
        entailment_threshold: float | None = None,
        groq_api_key: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.mode = mode or settings.verifier_mode
        self.embedding_model = embedding_model or get_embedding_model()
        self.nli_model_name = nli_model_name or settings.nli_model
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        self.entailment_threshold = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.nli_entailment_threshold
        )
        self.groq_api_key = groq_api_key if groq_api_key is not None else settings.groq_api_key
        self.llm_model = llm_model or settings.llm_model
        self._nli = None

    def _load_nli(self):
        if self._nli is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI model %s", self.nli_model_name)
            self._nli = CrossEncoder(self.nli_model_name, device=settings.embedding_device)
        return self._nli

    def _verify_similarity(self, claim: str, passage: str) -> tuple[bool, float, str]:
        # Encode both sides as documents (no retrieval query prefix).
        vectors = self.embedding_model.embed_documents([claim, passage])
        score = float(np.dot(vectors[0], vectors[1]))
        ok = score >= self.similarity_threshold
        return ok, score, "supported" if ok else "unsupported"

    def _nli_label_names(self, model) -> list[str]:
        labels = list(getattr(model, "labels", []) or [])
        if labels:
            return [str(x).lower() for x in labels]
        try:
            id2label = model.model.config.id2label
            return [str(id2label[i]).lower() for i in range(len(id2label))]
        except Exception:  # noqa: BLE001
            return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _verify_nli(self, claim: str, passage: str) -> tuple[bool, float, str]:
        model = self._load_nli()
        # Premise = passage, hypothesis = claim
        try:
            scores = model.predict([(passage, claim)], apply_softmax=True)
        except TypeError:
            scores = model.predict([(passage, claim)])
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        if scores.size == 1:
            entailment_prob = float(scores[0])
            if entailment_prob < 0 or entailment_prob > 1:
                entailment_prob = float(1 / (1 + np.exp(-entailment_prob)))
            ok = entailment_prob >= self.entailment_threshold
            return ok, entailment_prob, "supported" if ok else "unsupported"

        if scores.size == 3:
            # Some versions return logits even with apply_softmax; normalize if needed.
            if float(scores.min()) < 0.0 or float(scores.max()) > 1.0 or not np.isclose(scores.sum(), 1.0, atol=0.05):
                probs = self._softmax(scores)
            else:
                probs = scores
            labels = self._nli_label_names(model)
            if len(labels) != 3:
                labels = ["contradiction", "entailment", "neutral"]
            label_to_prob = {labels[i]: float(probs[i]) for i in range(3)}
            entailment = label_to_prob.get("entailment", 0.0)
            contradiction = label_to_prob.get("contradiction", 0.0)
            if entailment >= self.entailment_threshold:
                return True, entailment, "supported"
            if contradiction >= self.entailment_threshold:
                return False, contradiction, "contradicted"
            return False, entailment, "neutral"

        score = float(scores[0])
        ok = score >= self.entailment_threshold
        return ok, score, "supported" if ok else "unsupported"

    def _verify_llm(self, claim: str, passage: str) -> tuple[bool, float, str]:
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY missing for LLM judge; falling back to NLI")
            return self._verify_nli(claim, passage)

        from groq import Groq

        client = Groq(api_key=self.groq_api_key)
        prompt = (
            "Does the PASSAGE entail the CLAIM?\n"
            "Reply with exactly one word: YES, NO, or UNSURE.\n\n"
            f"PASSAGE:\n{passage}\n\nCLAIM:\n{claim}"
        )
        completion = client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (completion.choices[0].message.content or "").strip().upper()
        if text.startswith("YES"):
            return True, 1.0, "supported"
        if text.startswith("NO"):
            return False, 0.0, "unsupported"
        return False, 0.5, "unsure"

    def verify_citation(self, citation: Citation) -> Citation:
        claim = citation.claim or ""
        passage = citation.passage or ""
        if not claim.strip() or not passage.strip():
            citation.verified = False
            citation.verification_score = 0.0
            citation.verification_label = "missing"
            return citation

        if self.mode == "similarity":
            ok, score, label = self._verify_similarity(claim, passage)
        elif self.mode == "llm":
            ok, score, label = self._verify_llm(claim, passage)
        else:
            ok, score, label = self._verify_nli(claim, passage)

        citation.verified = ok
        citation.verification_score = score
        citation.verification_label = label
        return citation

    def verify(self, result: GenerationResult) -> GenerationResult:
        checked = [self.verify_citation(c) for c in result.citations]
        supported = [c for c in checked if c.verified]

        # Drop unsupported citations. If every citation fails, refuse to answer.
        if checked and not supported:
            result.answer = "I don't know based on the available documents."
            result.citations = []
        else:
            result.citations = supported
        return result


@lru_cache(maxsize=1)
def get_verifier() -> CitationVerifier:
    return CitationVerifier()
