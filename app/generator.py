"""LLM answer generation with grounded prompting (Groq)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from app.config import settings
from app.retriever import ScoredChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Use ONLY the provided context passages to answer.
If the answer is not in the context, reply exactly: I don't know based on the available documents.
Never invent facts, numbers, or policies.
Always cite sources using the citation markers already attached to each passage (e.g. [1], [2]).
Return a JSON object with this schema:
{
  "answer": "string",
  "citations": [
    {"marker": 1, "claim": "short claim supported by this source"}
  ]
}
Do not wrap the JSON in markdown fences."""


@dataclass
class Citation:
    marker: int
    claim: str
    source: str
    page: int
    chunk_id: str
    passage: str
    verified: bool | None = None
    verification_score: float | None = None
    verification_label: str | None = None


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    raw_response: str = ""
    context_used: list[dict] = field(default_factory=list)


def format_context(chunks: Sequence[ScoredChunk]) -> str:
    blocks: list[str] = []
    for i, scored in enumerate(chunks, start=1):
        c = scored.chunk
        blocks.append(
            f"[{i}] Source: {c.source} | Page: {c.page} | ID: {c.chunk_id}\n{c.content}"
        )
    return "\n\n".join(blocks)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class GroqGenerator:
    """Generate grounded answers via the Groq Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env file to enable generation."
                )
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                answer="I don't know based on the available documents.",
                citations=[],
                raw_response="",
                context_used=[],
            )

        context_block = format_context(contexts)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context passages:\n{context_block}\n\n"
            "Answer using only the context. Cite markers like [1]."
        )

        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = completion.choices[0].message.content or ""
        logger.debug("LLM raw response: %s", raw)

        try:
            payload = _extract_json(raw)
            answer = str(payload.get("answer", "")).strip()
            citation_items = payload.get("citations", []) or []
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Failed to parse JSON from LLM; using raw text")
            answer = raw.strip()
            citation_items = []
            # Infer markers mentioned in the answer.
            for marker in sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)}):
                citation_items.append({"marker": marker, "claim": answer})

        citations: list[Citation] = []
        for item in citation_items:
            try:
                marker = int(item.get("marker"))
            except (TypeError, ValueError, AttributeError):
                continue
            if marker < 1 or marker > len(contexts):
                continue
            scored = contexts[marker - 1]
            citations.append(
                Citation(
                    marker=marker,
                    claim=str(item.get("claim", "")).strip() or answer,
                    source=scored.chunk.source,
                    page=scored.chunk.page,
                    chunk_id=scored.chunk.chunk_id,
                    passage=scored.chunk.content,
                )
            )

        # If the model forgot citations but used markers, attach them.
        if not citations:
            for marker in sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)}):
                if 1 <= marker <= len(contexts):
                    scored = contexts[marker - 1]
                    citations.append(
                        Citation(
                            marker=marker,
                            claim=answer,
                            source=scored.chunk.source,
                            page=scored.chunk.page,
                            chunk_id=scored.chunk.chunk_id,
                            passage=scored.chunk.content,
                        )
                    )

        return GenerationResult(
            answer=answer or "I don't know based on the available documents.",
            citations=citations,
            raw_response=raw,
            context_used=[s.to_dict() for s in contexts],
        )
