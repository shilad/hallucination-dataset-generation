from pathlib import Path

import pytest

from hallucination_creation.claim_generator import ClaimDraft
from hallucination_creation.domains import DomainPrompt
from hallucination_creation.evaluator import EvaluationResult, SUPPORTED_VERDICT
from hallucination_creation.pipeline import DatasetPipeline, PipelineConfig
from hallucination_creation.retriever import EvidenceChunk


class FakeClaimGenerator:
    def __init__(self) -> None:
        self.generated = False

    def generate(self, domain: DomainPrompt):
        if self.generated:
            return []
        self.generated = True
        yield ClaimDraft(claim_text=f"{domain.name} claim", reasoning="mock reasoning")


class FakeRetriever:
    def retrieve(self, claim_text: str):
        yield EvidenceChunk(title="Example Source", url="https://example.com", snippet="Example snippet")


class FakeEvaluator:
    def evaluate(self, claim_text: str, evidence):
        return EvaluationResult(claim_text=claim_text, verdict=SUPPORTED_VERDICT, explanation="Mock explanation")


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
        config=PipelineConfig(max_claims_per_domain=1),
        claim_generator=FakeClaimGenerator(),
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(),
    )

    pipeline.run()

    raw_files = list((tmp_path / "raw").glob("claims_*.jsonl"))
    processed_files = list((tmp_path / "processed").glob("dataset_*.csv"))

    assert raw_files, "Expected a raw JSONL file to be written."
    assert processed_files, "Expected a processed CSV file to be written."
