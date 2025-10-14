"""Generate claims for hallucination evaluation across multiple domains."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .agent import _build_openai_model
from .domains import DomainPrompt


class ClaimDraft(BaseModel):
    """Structured output describing a generated claim."""

    claim_text: str = Field(..., description="Declarative statement to evaluate.")
    reasoning: Optional[str] = Field(
        None, description="Generator's chain-of-thought or supporting evidence."
    )


def _create_claim_agent(system_prompt: Optional[str] = None) -> Agent[ClaimDraft]:
    """Internal helper to build the claim generation agent."""

    base_prompt = (
        "You generate concise factual claims that can be validated via current web evidence. "
        "Focus on statements that are either true, false, or currently debated, so that a verification agent can assess them. "
        "Avoid trivia that is impossible to fact-check online. "
        "The dataset this feeds should be challenging for advanced evaluators, so prefer nuanced, high-impact claims with subtle details that require careful verification. "
        "Favor scenarios that demand reasoning about causal chains, timelines, or trade-offs rather than shallow recall. "
        "Only reference information that was true or reported on or before December 31, 2023."
    )

    composed_prompt = f"{system_prompt.strip()}\n{base_prompt}" if system_prompt else base_prompt

    return Agent(
        model=_build_openai_model(),
        output_type=ClaimDraft,
        system_prompt=composed_prompt,
    )


@dataclass
class ClaimGeneratorConfig:
    """Configuration values controlling claim generation."""

    max_attempts_per_domain: int = 3
    custom_system_prompt: Optional[str] = None


class ClaimGenerator:
    """Generate candidate claims given a domain description."""

    def __init__(self, *, config: Optional[ClaimGeneratorConfig] = None) -> None:
        self._config = config or ClaimGeneratorConfig()
        self._agent = _create_claim_agent(system_prompt=self._config.custom_system_prompt)

    async def generate_claim(
        self,
        domain: DomainPrompt,
        attempt: int,
        semaphore: asyncio.Semaphore,
    ) -> ClaimDraft:
        """Generate a single claim draft using the shared semaphore for LLM throttling."""

        prompt = self._build_prompt(domain, attempt)
        async with semaphore:
            result = await asyncio.to_thread(self._agent.run_sync, prompt)
        return result.output

    @staticmethod
    def _build_prompt(domain: DomainPrompt, attempt: int) -> str:
        """Compose the prompt that encourages diverse claim drafts."""

        return (
            f"Domain: {domain.name}\n"
            f"{domain.formatted_seed()}\n"
            f"Attempt {attempt + 1}: Generate one factual claim that is non-trivial and checkable, grounded in information reported on or before 2023-12-31.\n"
            "Ensure the claim requires multi-step reasoning or synthesis across sources rather than single data points. Provide the claim as `claim_text` and optionally include `reasoning`."
        )
