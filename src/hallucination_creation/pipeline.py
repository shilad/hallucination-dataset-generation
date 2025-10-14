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
from .writer import (
    DatasetWriter,
    ProcessedRecord,
    RawRecord,
    load_processed_records,
    load_raw_records,
)

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
        resume_raw_path: Optional[Path] = None,
        resume_processed_path: Optional[Path] = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._domains = list(domains or default_domains(self._config.max_domains))
        self._claim_generator = claim_generator or ClaimGenerator(config=self._config.claim_generator)
        self._retriever = retriever or OpenAIWebRetriever()
        self._evaluator = evaluator or HallucinationEvaluator()
        self._completed_counts: dict[str, int] = {}

        existing_processed: dict[tuple[str, str, str], ProcessedRecord] = {}

        if resume_raw_path is not None:
            raw_records = load_raw_records(resume_raw_path)
            for record in raw_records:
                self._completed_counts[record.domain] = self._completed_counts.get(record.domain, 0) + 1
                if record.processed_record is not None:
                    key = self._processed_key(record.processed_record)
                    existing_processed[key] = record.processed_record

            logger.info(
                "Loaded %s existing claims from %s for resume.",
                sum(self._completed_counts.values()),
                resume_raw_path,
            )

        if resume_processed_path is not None:
            processed_path = Path(resume_processed_path)
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed resume path {processed_path} does not exist.")
            for record in load_processed_records(processed_path):
                key = self._processed_key(record)
                existing_processed[key] = record

            logger.info("Loaded %s processed rows from %s.", len(existing_processed), processed_path)

        self._writer = DatasetWriter(
            data_root=data_root,
            balance_verdicts=self._config.balance_verdicts,
            resume_raw_path=resume_raw_path,
            existing_processed_records=list(existing_processed.values()),
        )
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

        max_claims = self._config.max_claims_per_domain
        if max_claims is None:
            raise ValueError("max_claims_per_domain must be specified when running the dataset pipeline.")

        completed = self._completed_counts.get(domain.name, 0)
        if completed >= max_claims:
            logger.info(
                "Domain %s already has %s/%s accepted claims from resume data. Skipping.",
                domain.name,
                completed,
                max_claims,
            )
            return

        attempt_index = completed

        while completed < max_claims:
            logger.info(
                "Generating claim %s/%s for domain %s",
                completed + 1,
                max_claims,
                domain.name,
            )

            draft = await self._claim_generator.generate_claim(domain, attempt_index, semaphore)
            attempt_index += 1

            evidence = list(
                await self._retriever.retrieve_async(draft.claim_text, semaphore)
            )

            evaluation = await self._evaluator.evaluate_async(draft.claim_text, evidence, semaphore)

            if not evaluation.is_clear:
                logger.info(
                    "Skipping ambiguous verdict for domain %s (accepted %s/%s so far).",
                    domain.name,
                    completed,
                    max_claims,
                )
                continue

            logger.info(
                "Verdict for domain %s claim %s/%s: %s",
                domain.name,
                completed + 1,
                max_claims,
                evaluation.verdict,
            )

            await self._persist_records(domain, draft, evaluation, evidence)
            completed += 1
            self._completed_counts[domain.name] = completed

    async def _persist_records(
        self,
        domain: DomainPrompt,
        draft: ClaimDraft,
        evaluation: EvaluationResult,
        evidence,
    ) -> None:
        """Write raw and processed records for a single claim evaluation."""

        collected_at = timestamp_now()
        prompt_text = self._build_prompt_text(domain, draft)
        serialized_context = serialize_evidence(evidence)
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
            processed_with_split = self._writer.append_processed(processed_record)
            stored_raw = RawRecord(
                domain=domain.name,
                prompt=prompt_text,
                claim_text=draft.claim_text,
                generator_reasoning=draft.reasoning,
                retrieved_context=serialized_context,
                collected_at=collected_at,
                processed_record=processed_with_split,
            )
            await asyncio.to_thread(self._writer.append_raw, stored_raw)

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

    @staticmethod
    def _processed_key(record: ProcessedRecord) -> tuple[str, str, str]:
        """Return a deduplication key for processed resume inputs."""

        return (record.domain, record.claim_text, record.collected_at)
