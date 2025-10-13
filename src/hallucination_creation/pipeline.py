"""End-to-end pipeline for hallucination dataset construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .claim_generator import ClaimDraft, ClaimGenerator, ClaimGeneratorConfig
from .domains import DomainPrompt, default_domains
from .evaluator import EvaluationResult, HallucinationEvaluator
from .retriever import EvidenceRetriever, OpenAIWebRetriever, serialize_evidence, timestamp_now
from .writer import DatasetWriter, ProcessedRecord, RawRecord

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Settings controlling the number of samples per domain."""

    max_claims_per_domain: int = 5
    claim_generator: Optional[ClaimGeneratorConfig] = None


class DatasetPipeline:
    """Coordinate generation, retrieval, evaluation, and persistence."""

    def __init__(
        self,
        *,
        data_root: Path,
        retriever: Optional[EvidenceRetriever] = None,
        domains: Optional[Iterable[DomainPrompt]] = None,
        config: Optional[PipelineConfig] = None,
        claim_generator: Optional[ClaimGenerator] = None,
        evaluator: Optional[HallucinationEvaluator] = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._domains = list(domains or default_domains())
        self._claim_generator = claim_generator or ClaimGenerator(config=self._config.claim_generator)
        self._retriever = retriever or OpenAIWebRetriever()
        self._evaluator = evaluator or HallucinationEvaluator()
        self._writer = DatasetWriter(data_root=data_root)

    def run(self) -> None:
        """Execute the pipeline for each configured domain."""

        for domain in self._domains:
            logger.info("Collecting claims for domain %s", domain.name)
            self._process_domain(domain)

        exported_path = self._writer.export_processed()
        logger.info("Processed dataset exported to %s", exported_path)
        logger.info("Raw interactions stored in %s", self._writer.raw_path)

    def _process_domain(self, domain: DomainPrompt) -> None:
        """Generate, validate, and store claims for a single domain."""

        for idx, draft in enumerate(self._claim_generator.generate(domain), start=1):
            if idx > self._config.max_claims_per_domain:
                break

            logger.debug("Evaluating claim %s/%s for domain %s", idx, self._config.max_claims_per_domain, domain.name)
            evidence = list(self._retriever.retrieve(draft.claim_text))
            evaluation = self._evaluator.evaluate(draft.claim_text, evidence)

            self._persist_records(domain, draft, evaluation, evidence)

    def _persist_records(
        self,
        domain: DomainPrompt,
        draft: ClaimDraft,
        evaluation: EvaluationResult,
        evidence,
    ) -> None:
        """Write raw and processed records for a single claim evaluation."""

        collected_at = timestamp_now()
        self._writer.append_raw(
            RawRecord(
                domain=domain.name,
                prompt=self._build_prompt_text(domain, draft),
                claim_text=draft.claim_text,
                generator_reasoning=draft.reasoning,
                retrieved_context=serialize_evidence(evidence),
                collected_at=collected_at,
            )
        )

        self._writer.append_processed(
            ProcessedRecord(
                claim_text=evaluation.claim_text,
                verdict=evaluation.verdict,
                explanation=evaluation.explanation,
                domain=domain.name,
                collected_at=collected_at,
            )
        )

    @staticmethod
    def _build_prompt_text(domain: DomainPrompt, draft: ClaimDraft) -> str:
        """Reconstruct the prompt that produced the claim for traceability."""

        return (
            f"Domain: {domain.name}\n"
            f"{domain.formatted_seed()}\n"
            "Instruction: Generate one factual claim suitable for verification."
        )
