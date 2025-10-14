"Incremental writers for raw and processed hallucination datasets."

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None


RAW_FILENAME_TEMPLATE = "claims_{timestamp}.jsonl"
PROCESSED_FILENAME_TEMPLATE = "dataset_{timestamp}"


@dataclass
class RawRecord:
    """Single raw interaction captured during data collection."""

    domain: str
    prompt: str
    claim_text: str
    generator_reasoning: Optional[str]
    retrieved_context: Iterable[dict]
    collected_at: str

    def to_json(self) -> str:
        payload = {
            "domain": self.domain,
            "prompt": self.prompt,
            "claim_text": self.claim_text,
            "generator_reasoning": self.generator_reasoning,
            "retrieved_context": list(self.retrieved_context),
            "collected_at": self.collected_at,
        }
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ProcessedRecord:
    """Curated dataset row ready for analysis."""

    claim_text: str
    verdict: str
    explanation: str
    domain: str
    collected_at: str
    split: Optional[str] = None
    reference_1: str = ""
    reference_2: str = ""
    reference_3: str = ""
    snippet_1: str = ""
    snippet_2: str = ""
    snippet_3: str = ""


@dataclass
class DatasetWriter:
    """Manage raw JSONL dumping and processed aggregation."""

    data_root: Path
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S"))
    processed_records: List[ProcessedRecord] = field(default_factory=list)
    _raw_file: Optional[Path] = field(init=False, default=None)
    balance_verdicts: bool = False

    def __post_init__(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)

    @property
    def raw_path(self) -> Path:
        if self._raw_file is None:
            raw_dir = self.data_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            self._raw_file = raw_dir / RAW_FILENAME_TEMPLATE.format(timestamp=self.timestamp)
        return self._raw_file

    @property
    def processed_dir(self) -> Path:
        processed = self.data_root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        return processed

    def append_raw(self, record: RawRecord) -> None:
        """Append a raw record to the JSON Lines artifact."""

        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json())
            handle.write("\n")

    def append_processed(self, record: ProcessedRecord) -> None:
        """Buffer processed records for later export."""

        assigned = replace(record, split=self._next_split())
        self.processed_records.append(assigned)

    def export_processed(self) -> Path:
        """Write the processed dataset to Parquet (or CSV fallback)."""

        if not self.processed_records:
            raise ValueError("No processed records to export.")

        records = self._balanced_records() if self.balance_verdicts else list(self.processed_records)

        if not records:
            raise ValueError("No processed records available for export.")

        base_name = PROCESSED_FILENAME_TEMPLATE.format(timestamp=self.timestamp)
        if pd is not None:
            return self._export_parquet(base_name, records)
        return self._export_csv(base_name, records)

    def _export_parquet(self, base_name: str, records: List[ProcessedRecord]) -> Path:
        """Export processed records to parquet using pandas."""

        file_path = self.processed_dir / f"{base_name}.parquet"
        df = pd.DataFrame([record.__dict__ for record in records])  # type: ignore[arg-type]
        df.to_parquet(file_path, index=False)
        return file_path

    def _export_csv(self, base_name: str, records: List[ProcessedRecord]) -> Path:
        """Fallback when pandas/pyarrow are unavailable."""

        file_path = self.processed_dir / f"{base_name}.csv"
        headers = [
            "claim_text",
            "verdict",
            "explanation",
            "domain",
            "collected_at",
            "split",
            "reference_1",
            "reference_2",
            "reference_3",
            "snippet_1",
            "snippet_2",
            "snippet_3",
        ]

        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow(record.__dict__)
        return file_path

    def _next_split(self) -> str:
        """Return `train` or `test` to maintain an 80/20 split."""

        index = len(self.processed_records) + 1
        return "test" if index % 5 in (0, 4) else "train"

    def _balanced_records(self) -> List[ProcessedRecord]:
        """Ensure the exported dataset is evenly balanced across verdict classes."""

        counts: dict[str, int] = {}
        for record in self.processed_records:
            counts[record.verdict] = counts.get(record.verdict, 0) + 1

        if len(counts) < 2:
            return []

        target = min(counts.values())
        if target == 0:
            return []

        kept: List[ProcessedRecord] = []
        used: dict[str, int] = {verdict: 0 for verdict in counts}

        for record in self.processed_records:
            verdict = record.verdict
            if used[verdict] >= target:
                continue
            kept.append(record)
            used[verdict] += 1

            if all(count >= target for count in used.values()):
                if len(kept) >= target * len(used):
                    break

        return kept
