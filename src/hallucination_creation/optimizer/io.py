from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

RawClaim = Dict[str, Any]


@dataclass
class DatasetSplits:
    train: List[RawClaim]
    test: List[RawClaim]


def discover_latest_processed(processed_dir: Path) -> Path:
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    candidates = sorted(processed_dir.glob("dataset_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed CSV files matching 'dataset_*.csv' in {processed_dir}"
        )
    return candidates[-1]


def resolve_processed_file(args: Mapping[str, Any]) -> Path:
    repo_root: Path = Path(args["repository_root"])  # argparse.Namespace compatible
    processed_dir = repo_root / "data" / "processed"
    p = args.get("processed_file")
    if p is not None:
        candidate = Path(p)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Provided processed file does not exist: {candidate}")
        return candidate
    return discover_latest_processed(processed_dir)


def build_processed_context(row: Dict[str, Any], mode: str = "refs_and_snippets") -> str:
    parts: List[str] = []
    for idx in range(1, 3 + 1):
        reference = (row.get(f"reference_{idx}") or "").strip()
        snippet = (row.get(f"snippet_{idx}") or "").strip()
        if not reference and not snippet:
            continue
        if mode == "snippets_only":
            segment = snippet
        elif mode == "refs_only":
            segment = reference
        else:  # refs_and_snippets (default)
            if reference and snippet:
                segment = f"{reference} — {snippet}"
            elif reference:
                segment = reference
            else:
                segment = snippet
        parts.append(segment)
    return "\n".join(parts) if parts else "<no context>"


def load_processed_claims(
    path: Path,
    *,
    limit: int | None = None,
    context_mode: str = "refs_and_snippets",
) -> tuple[List[RawClaim], Dict[str, str]]:
    records: List[RawClaim] = []
    label_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "claim_text",
            "verdict",
            "explanation",
            "domain",
            "collected_at",
            "reference_1",
            "reference_2",
            "reference_3",
            "snippet_1",
            "snippet_2",
            "snippet_3",
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Processed dataset {path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            claim_text = (row.get("claim_text") or "").strip()
            verdict_raw = row.get("verdict")
            if not claim_text:
                continue
            context = build_processed_context(row, mode=context_mode)
            record: RawClaim = {
                "claim_text": claim_text,
                "retrieved_context": context,
                "domain": row.get("domain", "unknown"),
                "explanation": (row.get("explanation") or "").strip(),
                "processed_verdict": _normalize_verdict(verdict_raw),
                "collected_at": row.get("collected_at"),
            }
            records.append(record)
            nv = record["processed_verdict"]
            if nv:
                label_map[claim_text] = nv
            if limit is not None and len(records) >= limit:
                break
    logging.info("Loaded %d processed claim(s) from %s", len(records), path)
    return records, label_map


def _normalize_verdict(verdict: str | None) -> str | None:
    if verdict is None:
        return None
    value = verdict.strip().lower()
    if value == "supported":
        return "true"
    if value == "unsupported":
        return "false"
    return None


def split_train_test(claims: List[RawClaim], test_size: int, *, rng: random.Random) -> DatasetSplits:
    if not claims:
        return DatasetSplits(train=[], test=[])
    indices = list(range(len(claims)))
    rng.shuffle(indices)
    test_size = max(0, min(test_size, len(indices)))
    test_indices = set(indices[:test_size])
    train_split: List[RawClaim] = []
    test_split: List[RawClaim] = []
    for idx, claim in enumerate(claims):
        (test_split if idx in test_indices else train_split).append(claim)
    logging.info("Split into %d train / %d test", len(train_split), len(test_split))
    return DatasetSplits(train=train_split, test=test_split)


def _normalize_context(context: Any) -> str:
    if context is None:
        return "<no context>"
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        lines: List[str] = []
        for idx, item in enumerate(context, start=1):
            if isinstance(item, dict):
                title = item.get("title") or "Untitled"
                url = item.get("url") or ""
                snippet = item.get("snippet") or ""
                parts = [f"[{idx}] {title}"]
                if url:
                    parts.append(f"URL: {url}")
                if snippet:
                    parts.append(f"Snippet: {snippet}")
                lines.append(" | ".join(parts))
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(context)


def convert_to_examples(claims: Sequence[RawClaim], label_lookup: Dict[str, str] | None = None):
    import dspy  # local import to keep import-time light

    examples: List[dspy.Example] = []
    for claim in claims:
        example = dspy.Example(
            claim_text=claim.get("claim_text", ""),
            retrieved_context=_normalize_context(claim.get("retrieved_context")),
            domain=claim.get("domain", "unknown"),
            explanation=claim.get("explanation", ""),
            processed_verdict=(label_lookup or {}).get(claim.get("claim_text", "")),
        )
        example = example.with_inputs("claim_text", "retrieved_context")
        examples.append(example)
    return examples


def summarize_claims(claims: Iterable[RawClaim]) -> None:
    claims = list(claims)
    if not claims:
        logging.warning("No claims available to summarize.")
        return
    domains = {claim.get("domain", "unknown") for claim in claims}
    with_context = sum(1 for claim in claims if claim.get("retrieved_context"))
    logging.info("Domains observed: %s", ", ".join(sorted(domains)))
    logging.info("Claims with retrieved context: %d/%d", with_context, len(claims))
