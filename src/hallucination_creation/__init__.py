"""Core package for hallucination control utilities built on Pydantic AI."""

from .agent import create_hallucination_agent
from .pipeline import DatasetPipeline, PipelineConfig
from .simple_evaluator import create_simple_evaluator

__all__ = [
    "create_hallucination_agent",
    "create_simple_evaluator",
    "DatasetPipeline",
    "PipelineConfig",
]
