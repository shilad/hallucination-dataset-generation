"""CLI wrapper around the scripted hallucination dataset pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from hallucination_creation import DatasetPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hallucination detection dataset samples.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directory where raw and processed datasets will be written.",
    )
    parser.add_argument(
        "--max-claims-per-domain",
        type=int,
        default=5,
        help="Maximum number of claims accepted per domain.",
    )
    parser.add_argument(
        "--max-parallel-llm",
        type=int,
        default=3,
        help="Maximum number of concurrent LLM calls (generation, retrieval, evaluation).",
    )
    parser.add_argument(
        "--balance-verdicts",
        action="store_true",
        help="Enable balancing of supported/unsupported verdict counts before export.",
    )
    parser.add_argument(
        "--num-domains",
        type=int,
        default=None,
        help="Limit the number of domains processed (defaults to all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    load_dotenv()

    pipeline = DatasetPipeline(
        data_root=Path(args.data_root),
        config=PipelineConfig(
            max_claims_per_domain=args.max_claims_per_domain,
            max_parallel_llm_calls=args.max_parallel_llm,
            balance_verdicts=args.balance_verdicts,
            max_domains=args.num_domains,
        ),
    )
    asyncio.run(pipeline.run_async())


if __name__ == "__main__":
    main()
