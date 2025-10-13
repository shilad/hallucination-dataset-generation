"""Web-backed evidence retrieval utilities."""

from __future__ import annotations

import json
import os
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

    def __init__(self, *, model: Optional[str] = None) -> None:
        self._client = OpenAI(api_key=self._require_key())
        self._model = model or os.getenv("OPENAI_RETRIEVER_MODEL", "gpt-4.1-mini")

    @staticmethod
    def _require_key() -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAIWebRetriever.")
        return api_key

    def retrieve(self, claim_text: str) -> Iterable[EvidenceChunk]:
        """Call the OpenAI Responses API with web search enabled and parse snippets."""

        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Use web search to gather recent evidence supporting or refuting the claim below. "
                        "Return a JSON object with an `evidence` array. Each entry should include "
                        "`title`, `url`, and `snippet` fields summarizing the finding.\n\n"
                        f"Claim: {claim_text}"
                    ),
                }
            ],
            tools=[{"type": "web_search"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "web_evidence",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                    "required": ["title", "url", "snippet"],
                                },
                            }
                        },
                        "required": ["evidence"],
                    },
                },
            },
        )

        return self._parse_chunks(response)

    @staticmethod
    def _parse_chunks(response) -> Iterable[EvidenceChunk]:
        """Normalize OpenAI response content into EvidenceChunk objects."""

        chunks: List[EvidenceChunk] = []
        outputs = getattr(response, "output", None) or []
        for item in outputs:
            content = getattr(item, "content", None) or []
            for block in content:
                if getattr(block, "type", None) != "output_text":
                    continue
                text = getattr(block, "text", "")
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
        return chunks


def serialize_evidence(chunks: Iterable[EvidenceChunk]) -> List[dict[str, str]]:
    """Convert evidence chunks to JSON-serializable dictionaries."""

    return [chunk.to_dict() for chunk in chunks]


def timestamp_now() -> str:
    """Return the current UTC timestamp for record keeping."""

    return datetime.now(tz=timezone.utc).isoformat()
