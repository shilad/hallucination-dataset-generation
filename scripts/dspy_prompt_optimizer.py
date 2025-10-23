#!/usr/bin/env python3
"""DSPy prompt optimizer — thin, readable CLI.

Flow:
- Load processed CSV (data/processed) → RawClaim records
- Split into train/test → DSPy Examples
- Choose program (simple/cot/dspy) and metric
- Optional GEPA optimization (only in dspy mode)
- Evaluate and print summaries; optional prompt preview
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

# Import streamlined helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hallucination_creation import optimizer as opt  # type: ignore


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="DSPy prompt optimization over hallucination claims")
    p.add_argument("--mode", choices=["simple", "cot", "dspy"], default="simple",
                   help="'simple' direct true/false, 'cot' short reasoning, 'dspy' enables GEPA optimization")
    p.add_argument("--repository-root", type=Path, default=default_root,
                   help="Path to repository root (defaults to script parent)")
    p.add_argument("--processed-file", type=Path,
                   help="Processed dataset (CSV). Defaults to latest dataset_*.csv in data/processed")
    p.add_argument("--max-records", type=int, default=None,
                   help="Optional cap on number of claims to ingest")
    p.add_argument("--test-size", type=int, default=20,
                   help="Number of examples in test split (default: 20)")
    p.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable GEPA optimization (only used in dspy mode)")
    p.add_argument("--max-gepa-evals", type=int, default=3,
                   help="Max full GEPA evaluations when in manual mode")
    p.add_argument("--gepa-mode", choices=["light", "medium", "heavy", "manual"], default="light",
                   help="GEPA search mode. 'manual' honors --max-gepa-evals; presets choose budgets")
    p.add_argument("--report-errors", action="store_true",
                   help="Log individual examples that fail the metric during evaluation")
    p.add_argument("--preview", action="store_true", help="Print first prompt preview for sanity check")
    p.add_argument("--dry-run", action="store_true", help="Do not call the LLM; only load data and (optionally) preview")
    p.add_argument("--model-name", default=opt.DEFAULT_MODEL,
                   help=f"OpenAI model for DSPy (default: {opt.DEFAULT_MODEL})")
    p.add_argument("--optimizer-model", default=opt.DEFAULT_OPTIMIZER_MODEL,
                   help=f"Model used for GEPA searches (default: {opt.DEFAULT_OPTIMIZER_MODEL})")
    p.add_argument("--use-rationale", action=argparse.BooleanOptionalAction, default=False,
                   help="When set, require the model to produce a rationale and score it alongside the verdict.")
    p.add_argument("--context-mode", choices=["refs_and_snippets", "snippets_only", "refs_only"],
                   default="refs_and_snippets", help="How to construct retrieved context text for prompts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling/shuffling")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging verbosity")
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")


def _resolve_processed_file(ns: argparse.Namespace) -> Path:
    return opt.resolve_processed_file({"repository_root": ns.repository_root, "processed_file": ns.processed_file})


def build_prompt_preview_example(claim: Dict[str, Any], mode: str, use_rationale: bool) -> str:
    if mode == "simple":
        template = (
            "You are assisting in a hallucination detection task.\n"
            "Claim:\n{claim_text}\n\n"
            "Retrieved context (may be empty):\n{retrieved_context}\n\n"
            "Respond only with the single word `true` if the claim is fully supported by the context, otherwise respond `false`."
        )
    elif use_rationale:
        template = (
            "You are assisting in a hallucination detection task.\n"
            "Claim:\n{claim_text}\n\n"
            "Retrieved context (may be empty):\n{retrieved_context}\n\n"
            "Produce a grounded rationale beginning with `CLEAR:` when evidence decisively supports or refutes the claim, or `UNCERTAIN:` otherwise.\n"
            "After the rationale, conclude with `Answer: true` if the claim is fully supported or `Answer: false` otherwise."
        )
    else:
        template = (
            "You are assisting in a hallucination detection task.\n"
            "Claim:\n{claim_text}\n\n"
            "Retrieved context (may be empty):\n{retrieved_context}\n\n"
            "Think through two short steps, then conclude with `Answer: true` if the claim is fully supported or `Answer: false` otherwise."
        )
    return template.format(
        claim_text=claim.get("claim_text", "<missing claim>"),
        retrieved_context=claim.get("retrieved_context", "<no context>"),
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    rng = random.Random(args.seed)

    if args.mode == "simple" and args.use_rationale:
        logging.error("--use-rationale is unavailable for simple mode outputs.")
        sys.exit(1)

    try:
        processed_path = _resolve_processed_file(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        sys.exit(1)

    claims, label_map = opt.load_processed_claims(
        processed_path,
        limit=args.max_records,
        context_mode=args.context_mode,
    )
    opt.summarize_claims(claims)

    splits = opt.split_train_test(claims, args.test_size, rng=rng)
    train_examples = opt.convert_to_examples(splits.train, label_map)
    test_examples = opt.convert_to_examples(splits.test, label_map)

    if not args.dry_run:
        try:
            import dspy  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("DSPy is required. Install with `pip install dspy-ai`.") from exc
        lm = opt.initialize_lm(args.model_name)
        dspy.settings.configure(lm=lm)

    has_labels = any(getattr(ex, "processed_verdict", None) for ex in train_examples + test_examples)
    program, metric_fn, verdict_attr, should_optimize = opt.choose_program_and_metric(
        mode=args.mode,
        has_labels=has_labels,
        optimize_flag=args.optimize,
        use_rationale=args.use_rationale,
    )

    optimizer_lm = None if args.dry_run else opt.initialize_optimizer_lm(args.optimizer_model)
    if optimizer_lm and args.mode != "dspy":
        logging.info("Optimizer model specified but non-dspy mode; ignoring.")
        optimizer_lm = None

    if not args.dry_run and should_optimize and train_examples:
        # Simple 80/20 fit/val split, deterministic by RNG
        idx = list(range(len(train_examples)))
        rng.shuffle(idx)
        val_n = max(1, int(len(idx) * 0.2)) if idx else 0
        val_set = set(idx[:val_n])
        fit_examples = [ex for i, ex in enumerate(train_examples) if i not in val_set]
        val_examples = [ex for i, ex in enumerate(train_examples) if i in val_set]
        program = opt.optimize_program(
            program,
            fit_examples,
            metric=metric_fn,
            max_evals=args.max_gepa_evals,
            gepa_mode=args.gepa_mode,
            optimizer_lm=optimizer_lm,
            val_examples=val_examples,
        )
    elif args.mode == "dspy" and not train_examples:
        logging.warning("No training examples for DSPy optimization; running baseline.")
    elif not args.dry_run and args.mode == "dspy":
        logging.info("Skipping optimization — using baseline prompt only.")

    if not args.dry_run and args.mode in {"simple", "cot"} and args.optimize:
        logging.info("Optimization only available in dspy mode; ignoring --optimize.")

    if args.preview and claims:
        preview_source = splits.test[0] if splits.test else splits.train[0]
        preview_text = build_prompt_preview_example(preview_source, args.mode, args.use_rationale)
        logging.info("Prompt preview:\n%s", preview_text)

    if args.dry_run:
        logging.info("Dry run complete — data loaded and prompt previewed (if requested).")
        return

    evaluation_examples = test_examples or train_examples
    artifacts = opt.evaluate_program(
        program,
        evaluation_examples,
        metric_fn=metric_fn,
        verdict_attr=verdict_attr,
        report_errors=args.report_errors,
        mode=args.mode,
    )

    logging.info("Final summary (%s mode): %s", args.mode, json.dumps(artifacts.metric_scores, indent=2))

    if label_map:
        accuracy_report = opt.compare_predictions_to_labels(artifacts.raw_predictions, label_map)
        logging.info("Label comparison (%s mode): %s", args.mode, json.dumps(accuracy_report, indent=2))


if __name__ == "__main__":
    main()
