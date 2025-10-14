import csv
import json
from pathlib import Path

import pytest

from hallucination_creation.claim_generator import ClaimDraft
from hallucination_creation.domains import DomainPrompt
from hallucination_creation.evaluator import (
    EvaluationResult,
    SUPPORTED_VERDICT,
    UNSUPPORTED_VERDICT,
)
from hallucination_creation.pipeline import DatasetPipeline, PipelineConfig
from hallucination_creation.retriever import EvidenceChunk


class FakeClaimGenerator:
    async def generate_claim(self, domain: DomainPrompt, attempt: int, semaphore) -> ClaimDraft:  # type: ignore[override]
        return ClaimDraft(claim_text=f"{domain.name} claim {attempt}", reasoning="mock reasoning")


class FakeRetriever:
    async def retrieve_async(self, claim_text: str, semaphore):  # type: ignore[override]
        return [EvidenceChunk(title="Example Source", url="https://example.com", snippet="Example snippet")]


class FakeEvaluator:
    def __init__(self) -> None:
        self.toggle = False

    async def evaluate_async(self, claim_text: str, evidence, semaphore):  # type: ignore[override]
        verdict = SUPPORTED_VERDICT if not self.toggle else UNSUPPORTED_VERDICT
        self.toggle = not self.toggle
        return EvaluationResult(
            claim_text=claim_text,
            verdict=verdict,
            explanation="CLEAR: Mock explanation",
            is_clear=True,
        )


@pytest.fixture()
def domain() -> DomainPrompt:
    return DomainPrompt(
        name="Test Domain",
        description="A domain used for tests.",
        seed_questions=["What is happening now?"],
    )


def test_pipeline_writes_raw_and_processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: DomainPrompt) -> None:
    # Force CSV export in case pandas is installed.
    import hallucination_creation.writer as writer  # type: ignore

    monkeypatch.setattr(writer, "pd", None)

    pipeline = DatasetPipeline(
        data_root=tmp_path,
        domains=[domain],
        config=PipelineConfig(max_claims_per_domain=2, balance_verdicts=True),
        claim_generator=FakeClaimGenerator(),
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(),
    )

    pipeline.run()

    raw_files = list((tmp_path / "raw").glob("claims_*.jsonl"))
    processed_files = list((tmp_path / "processed").glob("dataset_*.csv"))

    assert raw_files, "Expected a raw JSONL file to be written."
    assert processed_files, "Expected a processed CSV file to be written."

    # Verify the processed dataset includes the split column, remains balanced, and flags train/test.
    csv_path = processed_files[0]
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Processed dataset should contain rows."
    for row in rows:
        assert "split" in row, "Each row should include the 'split' column."
        for key in ("reference_1", "reference_2", "reference_3", "snippet_1", "snippet_2", "snippet_3"):
            assert key in row, f"Missing column {key}."
    assert rows[0]["split"] == "train", "First record should default to the train split."
    assert rows[0]["reference_1"], "reference_1 should capture the primary evidence."
    assert rows[0]["snippet_1"] in ("", "Example snippet"), "snippet_1 should reflect evidence text."

    verdict_counts = {}
    for row in rows:
        verdict = row["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    assert verdict_counts.get(SUPPORTED_VERDICT) == verdict_counts.get(UNSUPPORTED_VERDICT)


def test_pipeline_resume_from_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: DomainPrompt) -> None:
    # Force CSV usage to avoid pandas dependency.
    import hallucination_creation.writer as writer  # type: ignore

    monkeypatch.setattr(writer, "pd", None)

    first_pipeline = DatasetPipeline(
        data_root=tmp_path,
        domains=[domain],
        config=PipelineConfig(max_claims_per_domain=1, balance_verdicts=False),
        claim_generator=FakeClaimGenerator(),
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(),
    )
    first_pipeline.run()

    raw_files = list((tmp_path / "raw").glob("claims_*.jsonl"))
    processed_files = list((tmp_path / "processed").glob("dataset_*.csv"))

    assert len(raw_files) == 1
    assert len(processed_files) == 1

    resume_pipeline = DatasetPipeline(
        data_root=tmp_path,
        domains=[domain],
        config=PipelineConfig(max_claims_per_domain=3, balance_verdicts=False),
        claim_generator=FakeClaimGenerator(),
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(),
        resume_raw_path=raw_files[0],
        resume_processed_path=processed_files[0],
    )
    resume_pipeline.run()

    # Raw file should now contain three records total.
    with raw_files[0].open("r", encoding="utf-8") as handle:
        stored_rows = [json.loads(line) for line in handle if line.strip()]
    assert len(stored_rows) == 3
    assert all("processed_record" in row for row in stored_rows), "Raw resume entries should embed processed payloads."

    # Processed export should include the original and the two newly generated rows.
    with processed_files[0].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        processed_rows = list(reader)

    assert len(processed_rows) == 3
    assert {row["claim_text"] for row in processed_rows} == {
        "Test Domain claim 0",
        "Test Domain claim 1",
        "Test Domain claim 2",
    }
