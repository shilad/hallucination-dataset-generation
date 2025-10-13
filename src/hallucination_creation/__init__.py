"""Core package for hallucination control utilities built on Pydantic AI."""

from .agent import create_hallucination_agent
from .pipeline import DatasetPipeline, PipelineConfig

__all__ = ["create_hallucination_agent", "DatasetPipeline", "PipelineConfig"]
