import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
        help="Maximum number of accepted claims per domain.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Minimum log level for pipeline execution.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running the scripted hallucination dataset pipeline."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    load_dotenv()

    pipeline = DatasetPipeline(
        data_root=Path(args.data_root),
        config=PipelineConfig(max_claims_per_domain=args.max_claims_per_domain),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
