"""Lightweight evaluator built on GPT-4.1-mini without structured reasoning."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


class SimpleVerdict(BaseModel):
    """Minimal verdict schema returned by the simple evaluator."""

    verdict: str = Field(..., description="Either 'supported' or 'unsupported'.")


def create_simple_evaluator(system_prompt: Optional[str] = None) -> Agent[SimpleVerdict]:
    """Return an evaluator that maps claims to supported/unsupported without explanations."""

    base_prompt = (
        "Classify whether the given claim is factually supported by widely accepted public knowledge "
        "available on or before 2023-12-31. "
        "Respond with the single word 'supported' if the claim is reliable, otherwise respond with "
        "'unsupported'. Do not add explanations or extra words."
    )

    composed_prompt = f"{system_prompt.strip()}\n{base_prompt}" if system_prompt else base_prompt

    agent = Agent(
        model=OpenAIChatModel("gpt-4.1-mini"),
        output_type=SimpleVerdict,
        system_prompt=composed_prompt,
    )
    return agent
