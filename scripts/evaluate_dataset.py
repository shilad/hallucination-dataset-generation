"""Evaluate the processed dataset with a lightweight GPT-4.1-mini classifier."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import dotenv

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hallucination_creation.simple_evaluator import create_simple_evaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate processed dataset with a simple GPT-4.1-mini classifier.")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to the processed dataset (CSV). Defaults to the most recent file in data/processed/.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-parallel-llm",
        type=int,
        default=3,
        help="Maximum number of concurrent LLM calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows evaluated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def find_latest_dataset() -> Path:
    processed_dir = Path("data/processed")
    candidates = sorted(processed_dir.glob("dataset_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No processed dataset found in data/processed/.")
    return candidates[0]


def load_rows(path: Path, split: str, limit: Optional[int]) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[dict] = []
        for row in reader:
            if split != "all" and row.get("split") != split:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows found for split='{split}' in {path}.")
    return rows


async def evaluate_rows(rows: Iterable[dict], max_parallel: int) -> List[str]:
    agent = create_simple_evaluator()
    semaphore = asyncio.Semaphore(max_parallel)

    async def classify(row: dict) -> str:
        prompt = f"Claim: {row['claim_text']}"
        async with semaphore:
            result = await asyncio.to_thread(agent.run_sync, prompt)
        return result.output.verdict.lower().strip()

    tasks = [asyncio.create_task(classify(row)) for row in rows]
    return await asyncio.gather(*tasks)


def summarize(rows: List[dict], predictions: List[str]) -> None:
    gold = [row["verdict"].lower().strip() for row in rows]
    total = len(rows)
    correct = sum(1 for g, p in zip(gold, predictions) if g == p)

    logging.info("Evaluated %s rows; accuracy %.2f%%", total, (correct / total) * 100)

    counts = Counter(predictions)
    logging.info("Prediction distribution: %s", dict(counts))

    confusion = Counter((g, p) for g, p in zip(gold, predictions))
    logging.info("Confusion matrix (gold, predicted): %s", dict(confusion))


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    dotenv.load_dotenv()

    dataset_path = args.dataset or find_latest_dataset()
    logging.info("Using dataset: %s", dataset_path)

    rows = load_rows(dataset_path, args.split, args.limit)
    logging.info("Loaded %s rows for split '%s'", len(rows), args.split)

    predictions = await evaluate_rows(rows, args.max_parallel_llm)
    summarize(rows, predictions)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
