"""Hallucination evaluation pipeline components."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .agent import HallucinationAssessment, create_hallucination_agent
from .retriever import EvidenceChunk

SUPPORTED_VERDICT = "supported"
UNSUPPORTED_VERDICT = "unsupported"
MIXED_VERDICT = "mixed"


@dataclass
class EvaluationResult:
    """Standardized structure for processed dataset rows."""

    claim_text: str
    verdict: str
    explanation: str
    is_clear: bool


class HallucinationEvaluator:
    """Evaluate claims using structured prompting and retrieved evidence."""

    def __init__(
        self,
        *,
        extra_guidelines: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._agent = create_hallucination_agent(
            system_prompt=system_prompt,
            extra_guidelines=extra_guidelines,
        )

    async def evaluate_async(
        self,
        claim_text: str,
        evidence: Iterable[EvidenceChunk],
        semaphore: asyncio.Semaphore,
    ) -> EvaluationResult:
        """Return the verdict and explanation for a claim given web evidence."""

        evidence_summary = self._summarize_evidence(evidence)
        prompt = self._build_prompt(claim_text, evidence_summary)

        async with semaphore:
            result = await asyncio.to_thread(self._agent.run_sync, prompt)
        assessment: HallucinationAssessment = result.output
        verdict = SUPPORTED_VERDICT if assessment.is_supported else UNSUPPORTED_VERDICT
        reasoning = assessment.reasoning.strip()
        is_clear = reasoning.upper().startswith("CLEAR:")

        return EvaluationResult(
            claim_text=claim_text,
            verdict=verdict,
            explanation=assessment.reasoning,
            is_clear=is_clear,
        )

    @staticmethod
    def _summarize_evidence(chunks: Iterable[EvidenceChunk]) -> str:
        """Combine evidence snippets into a single string for the agent."""

        summaries: List[str] = []
        for chunk in chunks:
            snippet = chunk.snippet.strip()
            if not snippet:
                continue
            title = chunk.title.strip() or "Untitled Source"
            summaries.append(f"{title}: {snippet} (Source: {chunk.url})")
        return "\n".join(summaries) if summaries else "No supporting evidence retrieved."

    @staticmethod
    def _build_prompt(claim_text: str, evidence_summary: str) -> str:
        """Construct the user prompt sent to the hallucination agent."""

        return (
            "Evaluate the factual accuracy of the given claim.\n"
            f"Claim: {claim_text}\n"
            f"Evidence:\n{evidence_summary}\n"
            "Return your reasoning and set is_supported to False when evidence contradicts or fails to verify the claim."
        )
