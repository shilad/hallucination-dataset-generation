"""Helpers for constructing Pydantic AI agents used in hallucination detection workflows."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


class HallucinationAssessment(BaseModel):
    """Structured response produced by the hallucination evaluation agent."""

    reasoning: str = Field(..., description="Explanation describing the evidence for the judgment.")
    is_supported: bool = Field(..., description="True when the claim is factually supported.")


def create_hallucination_agent(
    system_prompt: Optional[str] = None,
    *,
    extra_guidelines: Optional[Iterable[str]] = None,
) -> Agent[HallucinationAssessment]:
    """Create an agent that evaluates claims for hallucination risk."""

    prompt_lines = [
        "Assess whether the provided statement is factually grounded.",
        "If any part of the statement cannot be verified, mark is_supported as False.",
        "Use the supplied evidence summary to ground your reasoning.",
        "Always populate reasoning with citations or the information gap you identify.",
    ]

    if extra_guidelines:
        prompt_lines.extend(extra_guidelines)

    base_prompt = "\n".join(prompt_lines)
    composed_prompt = f"{system_prompt.strip()}\n{base_prompt}" if system_prompt else base_prompt

    agent = Agent(
        model=_build_openai_model(),
        output_type=HallucinationAssessment,
        system_prompt=composed_prompt,
    )
    return agent


def _build_openai_model() -> OpenAIChatModel:
    """Instantiate the OpenAI chat model, requiring the API key to be available."""

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        return OpenAIChatModel(model_name)

    raise RuntimeError(
        "OPENAI_API_KEY must be set before creating the hallucination agent. "
        "Copy .env.example to .env and supply a valid key."
    )
