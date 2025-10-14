"""Web-backed evidence retrieval utilities."""

from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from openai import OpenAI


@dataclass
class EvidenceChunk:
    """Represents an evidence snippet retrieved from the web."""

    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class EvidenceRetriever:
    """Abstract base class for web retrievers."""

    def retrieve(self, claim_text: str) -> Iterable[EvidenceChunk]:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIWebRetriever(EvidenceRetriever):
    """Implementation that leverages OpenAI's web search tool via the Responses API."""

    def __init__(self, *, model: Optional[str] = None, reasoning_effort: str = "medium") -> None:
        self._client = OpenAI(api_key=self._require_key())
        self._model = model or os.getenv("OPENAI_RETRIEVER_MODEL", "gpt-5")
        self._reasoning_effort = os.getenv("OPENAI_RETRIEVER_REASONING_EFFORT", reasoning_effort)

    @staticmethod
    def _require_key() -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAIWebRetriever.")
        return api_key

    async def retrieve_async(
        self,
        claim_text: str,
        semaphore: asyncio.Semaphore,
    ) -> Iterable[EvidenceChunk]:
        """Call the OpenAI Responses API with web search enabled and parse snippets."""

        instruction = (
            "Use available web search tools to gather evidence supporting or refuting the claim below. "
            "Respond with a JSON object containing an `evidence` array. Each entry must have `title`, `url`, "
            "and `snippet` fields summarizing one relevant source. "
            "Only include evidence published on or before 2023-12-31; ignore later developments.\n\n"
            f"Claim: {claim_text}"
        )

        async with semaphore:
            response = await asyncio.to_thread(
                self._client.responses.create,
                model=self._model,
                input=instruction,
                tools=[{"type": "web_search"}],
                reasoning={"effort": self._reasoning_effort},
            )

        return self._parse_chunks(response)

    @staticmethod
    def _parse_chunks(response) -> Iterable[EvidenceChunk]:
        """Normalize OpenAI response content into EvidenceChunk objects."""

        chunks: List[EvidenceChunk] = []
        texts: List[str] = []

        aggregated = getattr(response, "output_text", None)
        if aggregated:
            texts.append(aggregated)

        outputs = getattr(response, "output", None) or []
        for item in outputs:
            content = getattr(item, "content", None) or []
            for block in content:
                if getattr(block, "type", None) != "output_text":
                    continue
                text = getattr(block, "text", "")
                if text:
                    texts.append(text)

        for text in texts:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            for entry in data.get("evidence", []):
                chunk = EvidenceChunk(
                    title=entry.get("title", "Unknown Title"),
                    url=entry.get("url", ""),
                    snippet=entry.get("snippet", ""),
                )
                chunks.append(chunk)

        if not chunks and texts:
            chunks.append(EvidenceChunk(title="Web Summary", url="", snippet=texts[0]))

        return chunks


def serialize_evidence(chunks: Iterable[EvidenceChunk]) -> List[dict[str, str]]:
    """Convert evidence chunks to JSON-serializable dictionaries."""

    return [chunk.to_dict() for chunk in chunks]


def timestamp_now() -> str:
    """Return the current UTC timestamp for record keeping."""

    return datetime.now(tz=timezone.utc).isoformat()
