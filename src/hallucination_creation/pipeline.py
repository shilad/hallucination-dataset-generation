"""End-to-end pipeline for hallucination dataset construction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .claim_generator import ClaimDraft, ClaimGenerator, ClaimGeneratorConfig
from .domains import ALL_DOMAINS, DomainPrompt, default_domains
from .evaluator import EvaluationResult, HallucinationEvaluator
from .retriever import EvidenceRetriever, OpenAIWebRetriever, serialize_evidence, timestamp_now
from .writer import DatasetWriter, ProcessedRecord, RawRecord

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Settings controlling the number of samples per domain."""

    max_claims_per_domain: int = 5
    claim_generator: Optional[ClaimGeneratorConfig] = None
    max_parallel_llm_calls: int = 3
    balance_verdicts: bool = False
    max_domains: Optional[int] = None


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
        self._domains = list(domains or default_domains(self._config.max_domains))
        self._claim_generator = claim_generator or ClaimGenerator(config=self._config.claim_generator)
        self._retriever = retriever or OpenAIWebRetriever()
        self._evaluator = evaluator or HallucinationEvaluator()
        self._writer = DatasetWriter(data_root=data_root, balance_verdicts=self._config.balance_verdicts)
        self._writer_lock: Optional[asyncio.Lock] = None

    async def run_async(self) -> None:
        """Execute the pipeline for each configured domain asynchronously."""

        semaphore = asyncio.Semaphore(self._config.max_parallel_llm_calls)
        self._writer_lock = asyncio.Lock()

        tasks = [asyncio.create_task(self._process_domain(domain, semaphore)) for domain in self._domains]
        await asyncio.gather(*tasks)

        exported_path = self._writer.export_processed()
        logger.info("Processed dataset exported to %s", exported_path)
        logger.info("Raw interactions stored in %s", self._writer.raw_path)

    def run(self) -> None:
        """Synchronous wrapper around the async pipeline run."""

        asyncio.run(self.run_async())

    async def _process_domain(self, domain: DomainPrompt, semaphore: asyncio.Semaphore) -> None:
        """Generate, validate, and store claims for a single domain."""

        logger.info("Collecting claims for domain %s", domain.name)

        for attempt in range(self._config.max_claims_per_domain):
            idx = attempt + 1

            logger.info(
                "Generating claim %s/%s for domain %s",
                idx,
                self._config.max_claims_per_domain,
                domain.name,
            )

            draft = await self._claim_generator.generate_claim(domain, attempt, semaphore)

            evidence = list(
                await self._retriever.retrieve_async(draft.claim_text, semaphore)
            )

            evaluation = await self._evaluator.evaluate_async(draft.claim_text, evidence, semaphore)

            if not evaluation.is_clear:
                logger.info(
                    "Skipping ambiguous verdict for domain %s claim %s/%s",
                    domain.name,
                    idx,
                    self._config.max_claims_per_domain,
                )
                continue

            logger.info(
                "Verdict for domain %s claim %s/%s: %s",
                domain.name,
                idx,
                self._config.max_claims_per_domain,
                evaluation.verdict,
            )

            await self._persist_records(domain, draft, evaluation, evidence)

    async def _persist_records(
        self,
        domain: DomainPrompt,
        draft: ClaimDraft,
        evaluation: EvaluationResult,
        evidence,
    ) -> None:
        """Write raw and processed records for a single claim evaluation."""

        collected_at = timestamp_now()
        raw_record = RawRecord(
            domain=domain.name,
            prompt=self._build_prompt_text(domain, draft),
            claim_text=draft.claim_text,
            generator_reasoning=draft.reasoning,
            retrieved_context=serialize_evidence(evidence),
            collected_at=collected_at,
        )

        reference_slots, snippet_slots = self._extract_top_evidence(evidence)

        processed_record = ProcessedRecord(
            claim_text=evaluation.claim_text,
            verdict=evaluation.verdict,
            explanation=evaluation.explanation,
            domain=domain.name,
            collected_at=collected_at,
            reference_1=reference_slots[0],
            reference_2=reference_slots[1],
            reference_3=reference_slots[2],
            snippet_1=snippet_slots[0],
            snippet_2=snippet_slots[1],
            snippet_3=snippet_slots[2],
        )

        if self._writer_lock is None:
            raise RuntimeError("Writer lock not initialized; run the pipeline via run_async().")

        async with self._writer_lock:
            await asyncio.to_thread(self._writer.append_raw, raw_record)
            self._writer.append_processed(processed_record)

    @staticmethod
    def _build_prompt_text(domain: DomainPrompt, draft: ClaimDraft) -> str:
        """Reconstruct the prompt that produced the claim for traceability."""

        return (
            f"Domain: {domain.name}\n"
            f"{domain.formatted_seed()}\n"
            "Instruction: Generate one factual claim suitable for verification. Make it nuanced enough to challenge an expert fact-checker, and limit context to information available on or before 2023-12-31."
        )

    @staticmethod
    def _extract_top_evidence(evidence) -> tuple[list[str], list[str]]:
        """Return up to three reference strings and snippets from evidence chunks."""

        references: list[str] = []
        snippets: list[str] = []

        for chunk in evidence:
            title = (chunk.title or "Untitled Source").strip()
            url = (chunk.url or "").strip()
            snippet = (chunk.snippet or "").strip()

            if title and url:
                references.append(f"{title} ({url})")
            elif url:
                references.append(url)
            else:
                references.append(title)

            snippets.append(snippet)

            if len(references) >= 3:
                break

        while len(references) < 3:
            references.append("")
        while len(snippets) < 3:
            snippets.append("")

        return references, snippets
