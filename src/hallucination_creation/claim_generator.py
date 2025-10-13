"""Generate claims for hallucination evaluation across multiple domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

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
        "Avoid trivia that is impossible to fact-check online."
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

    def generate(self, domain: DomainPrompt) -> Iterable[ClaimDraft]:
        """Yield up to `max_attempts_per_domain` claim drafts for the domain."""

        for attempt in range(self._config.max_attempts_per_domain):
            prompt = self._build_prompt(domain, attempt)
            result = self._agent.run_sync(prompt)
            yield result.output

    @staticmethod
    def _build_prompt(domain: DomainPrompt, attempt: int) -> str:
        """Compose the prompt that encourages diverse claim drafts."""

        return (
            f"Domain: {domain.name}\n"
            f"{domain.formatted_seed()}\n"
            f"Attempt {attempt + 1}: Generate one factual claim that is non-trivial and checkable.\n"
            "Provide the claim as `claim_text` and optionally include `reasoning`."
        )
