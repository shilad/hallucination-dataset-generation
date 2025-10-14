#!/usr/bin/env python3
"""Command-line helper for DSPy prompt optimization experiments.

By default the script now consumes the processed CSV exports in ``data/processed``
so we can train and evaluate against finalized verdict labels while iterating on
prompt quality with OpenAI's ``gpt-4.1-mini``. You can optionally specify a
different LM (e.g., ``openai/gpt-5``) for DSPy teleprompting via
``--optimizer-model``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

try:
    import dspy  # type: ignore
    from dspy.teleprompt import GEPA
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
except ImportError as exc:  # pragma: no cover - fail fast if DSPy missing.
    raise SystemExit(
        "DSPy is required for this script. Install it with `pip install dspy-ai`."
    ) from exc


RawClaim = Dict[str, Any]

TRUE_FALSE_PATTERN = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def normalize_verdict(verdict: str | None) -> str | None:
    """Convert dataset verdict labels into normalized boolean strings."""

    if verdict is None:
        return None

    value = verdict.strip().lower()
    if value == "supported":
        return "true"
    if value == "unsupported":
        return "false"
    return None


def reasoning_lm_kwargs(model_identifier: str, effort: str | None = None) -> Dict[str, Any]:
    """Return overrides for reasoning models (temperature, tokens, effort)."""

    base = model_identifier.split("/")[-1].lower()
    if re.match(r"^(?:o[1345]|gpt-5)(?:-[a-z]+)?", base):
        kwargs: Dict[str, Any] = {"temperature": 1.0, "max_tokens": 16000}
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs
    return {}


@dataclass
class DatasetSplits:
    """Simple container for train/test splits."""

    train: List[RawClaim]
    test: List[RawClaim]


@dataclass
class DSPyArtifacts:
    """Holds DSPy objects created during optimization for downstream use."""

    program: dspy.Module
    metric_scores: Dict[str, float]
    raw_predictions: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the prompt optimizer script."""

    default_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Scaffold DSPy prompt optimization over hallucination claims.",
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "cot", "dspy"],
        default="simple",
        help=(
            "Classification mode: 'simple' for direct true/false, 'cot' for a brief chain-of-thought, "
            "'dspy' for DSPy-guided optimization (default: simple)."
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=default_root,
        help="Path to the repository root (defaults to script parent).",
    )
    parser.add_argument(
        "--processed-file",
        type=Path,
        help=(
            "Processed dataset (CSV) to load. Defaults to the most recent file in data/processed."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of raw claims to ingest.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=20,
        help="Number of examples to reserve for test evaluation (default: 20).",
    )
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the DSPy teleprompter (GEPA) to optimize prompts (default: enabled).",
    )
    parser.add_argument(
        "--max-gepa-evals",
        type=int,
        default=3,
        help="Maximum full GEPA evaluations when running in manual mode.",
    )
    parser.add_argument(
        "--gepa-mode",
        choices=["light", "medium", "heavy", "manual"],
        default="light",
        help=(
            "GEPA search mode. Use 'manual' to honor --max-gepa-evals; the presets choose their own budgets."
        ),
    )
    parser.add_argument(
        "--report-errors",
        action="store_true",
        help="Log individual examples that fail the metric during evaluation.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a preview of the first prompt template fill for sanity checking.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4.1-mini",
        help="OpenAI model name configured through DSPy (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--optimizer-model",
        default="openai/gpt-5-mini",
        help="Model used for DSPy teleprompter searches (default: openai/gpt-5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for any sampling or shuffling steps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the script run.",
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure root logger with a consistent format."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def discover_latest_processed(processed_dir: Path) -> Path:
    """Return the newest processed CSV dataset within ``processed_dir``."""

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    candidates = sorted(processed_dir.glob("dataset_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed CSV files matching 'dataset_*.csv' in {processed_dir}"
        )

    return candidates[-1]


def resolve_processed_file(args: argparse.Namespace) -> Path:
    """Resolve which processed CSV dataset to load based on CLI arguments."""

    repo_root: Path = args.repository_root
    processed_dir = repo_root / "data" / "processed"

    if args.processed_file is not None:
        candidate = args.processed_file
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Provided processed file does not exist: {candidate}")
        return candidate

    return discover_latest_processed(processed_dir)


def load_raw_claims(path: Path, limit: int | None = None) -> List[RawClaim]:
    """Load raw claim rows from ``path`` (JSON Lines format)."""

    records: List[RawClaim] = []
    if limit is not None and limit <= 0:
        logging.info("Limit set to %s; skipping load from %s", limit, path)
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSON on line %d: %s", line_number, exc)
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break

    logging.info("Loaded %d raw claim(s) from %s", len(records), path)
    return records


def build_processed_context(row: Dict[str, Any]) -> str:
    """Assemble evidence strings from reference/snippet columns only."""

    parts: List[str] = []

    for idx in range(1, 4):
        reference = (row.get(f"reference_{idx}") or "").strip()
        snippet = (row.get(f"snippet_{idx}") or "").strip()
        if not reference and not snippet:
            continue

        if reference and snippet:
            segment = f"{reference} — {snippet}"
        elif reference:
            segment = reference
        else:
            segment = snippet
        parts.append(segment)

    return "\n".join(parts) if parts else "<no context>"


def load_processed_claims(
    path: Path, limit: int | None = None
) -> tuple[List[RawClaim], Dict[str, str]]:
    """Load processed claim rows from ``path`` (CSV format) and label map."""

    records: List[RawClaim] = []
    label_map: Dict[str, str] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"claim_text", "verdict"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Processed dataset {path} is missing columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            claim_text = (row.get("claim_text") or "").strip()
            verdict_raw = row.get("verdict")
            normalized_verdict = normalize_verdict(verdict_raw)

            if not claim_text:
                continue

            context = build_processed_context(row)
            record: RawClaim = {
                "claim_text": claim_text,
                "retrieved_context": context,
                "domain": row.get("domain", "unknown"),
                "processed_verdict": normalized_verdict,
                "collected_at": row.get("collected_at"),
            }
            records.append(record)
            if normalized_verdict:
                label_map[claim_text] = normalized_verdict

            if limit is not None and len(records) >= limit:
                break

    logging.info("Loaded %d processed claim(s) from %s", len(records), path)
    return records, label_map


def split_train_test(
    claims: List[RawClaim], test_size: int, *, rng: random.Random
) -> DatasetSplits:
    """Split ``claims`` into train/test subsets with reproducible shuffling."""

    if not claims:
        return DatasetSplits(train=[], test=[])

    indices = list(range(len(claims)))
    rng.shuffle(indices)

    actual_test = min(max(test_size, 0), len(indices))
    test_indices = set(indices[:actual_test])

    train_split: List[RawClaim] = []
    test_split: List[RawClaim] = []

    for idx, claim in enumerate(claims):
        if idx in test_indices:
            test_split.append(claim)
        else:
            train_split.append(claim)

    logging.info(
        "Split claims into %d train and %d test examples (requested test size=%d)",
        len(train_split),
        len(test_split),
        test_size,
    )

    return DatasetSplits(train=train_split, test=test_split)


def split_fit_validation(
    examples: Sequence[dspy.Example],
    val_fraction: float,
    *,
    rng: random.Random,
) -> tuple[List[dspy.Example], List[dspy.Example]]:
    """Split examples into fit and validation subsets."""

    if not examples:
        return [], []

    indices = list(range(len(examples)))
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_fraction))
    if val_count >= len(indices):
        val_count = len(indices) // 2 or 1

    val_indices = set(indices[:val_count])
    fit_examples: List[dspy.Example] = []
    val_examples: List[dspy.Example] = []

    for idx, example in enumerate(examples):
        if idx in val_indices:
            val_examples.append(example)
        else:
            fit_examples.append(example)

    if not fit_examples:
        fit_examples, val_examples = val_examples, fit_examples

    return fit_examples, val_examples


def summarize_claims(claims: Iterable[RawClaim]) -> None:
    """Log a short summary of the loaded claim set for quick inspection."""

    claims = list(claims)
    if not claims:
        logging.warning("No claims available to summarize.")
        return

    domains = {claim.get("domain", "unknown") for claim in claims}
    with_context = sum(1 for claim in claims if claim.get("retrieved_context"))

    logging.info("Domains observed: %s", ", ".join(sorted(domains)))
    logging.info("Claims with retrieved context: %d/%d", with_context, len(claims))


def initialize_lm(model_name: str) -> dspy.clients.LM:
    """Initialize the DSPy language model wrapper for OpenAI."""

    logging.debug("Initializing DSPy LM wrapper with model '%s'", model_name)
    if "/" in model_name:
        provider_model = model_name
    else:
        provider_model = f"openai/{model_name}"

    kwargs = reasoning_lm_kwargs(provider_model)
    return dspy.clients.LM(model=provider_model, model_type="chat", **kwargs)


def initialize_optimizer_lm(model_name: str | None) -> dspy.clients.LM | None:
    """Initialize a separate LM for optimization, if requested."""

    if not model_name:
        return None

    logging.debug("Initializing optimizer LM with model '%s'", model_name)
    if "/" in model_name:
        provider_model = model_name
    else:
        provider_model = f"openai/{model_name}"

    kwargs = reasoning_lm_kwargs(provider_model, effort="low")
    return dspy.clients.LM(model=provider_model, model_type="chat", **kwargs)


def build_prompt_preview_example(claim: RawClaim, mode: str) -> str:
    """Generate a prompt preview for the first claim based on the selected mode."""

    if mode == "cot":
        template = (
            "You are assisting in a hallucination detection task.\n"
            "Claim:\n{claim_text}\n\n"
            "Retrieved context (may be empty):\n{retrieved_context}\n\n"
            "Think through two short steps, then conclude with `Answer: true` if the claim is fully supported or `Answer: false` otherwise."
        )
    else:
        template = (
            "You are assisting in a hallucination detection task.\n"
            "Claim:\n{claim_text}\n\n"
            "Retrieved context (may be empty):\n{retrieved_context}\n\n"
            "Respond only with the single word `true` if the claim is fully supported by the context, otherwise respond `false`."
        )

    return template.format(
        claim_text=claim.get("claim_text", "<missing claim>"),
        retrieved_context=normalize_context(claim.get("retrieved_context")),
    )


def normalize_context(context: Any) -> str:
    """Convert the ``retrieved_context`` field into a human-readable string."""

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


def convert_to_examples(
    claims: Sequence[RawClaim],
    label_lookup: Dict[str, str] | None = None,
) -> List[dspy.Example]:
    """Transform claim dictionaries into DSPy ``Example`` objects."""

    examples: List[dspy.Example] = []
    for claim in claims:
        example = dspy.Example(
            claim_text=claim.get("claim_text", ""),
            retrieved_context=normalize_context(claim.get("retrieved_context")),
            domain=claim.get("domain", "unknown"),
            processed_verdict=claim.get("processed_verdict")
            or (label_lookup.get(claim.get("claim_text", "")) if label_lookup else None),
        ).with_inputs("claim_text", "retrieved_context")
        examples.append(example)

    return examples


def normalize_verdict_text(text: str) -> str:
    """Extract a normalized boolean verdict (`true` or `false`) from text."""

    if not text:
        return ""

    match = TRUE_FALSE_PATTERN.search(text)
    if not match:
        return ""

    return match.group(1).lower()


def load_processed_verdicts(path: Path, repo_root: Path) -> Dict[str, str]:
    """Load ground-truth verdicts from a processed CSV dataset."""

    if not path.is_absolute():
        path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV processed datasets are supported at this time.")

    verdict_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "claim_text" not in reader.fieldnames or "verdict" not in reader.fieldnames:
            raise ValueError(
                "Processed dataset must include 'claim_text' and 'verdict' columns."
            )

        for row in reader:
            claim = (row.get("claim_text") or "").strip()
            verdict = (row.get("verdict") or "").strip().lower()
            if not claim or not verdict:
                continue
            normalized = normalize_verdict(verdict)
            if normalized:
                verdict_map[claim] = normalized

    logging.info(
        "Loaded %d labeled verdict(s) from processed dataset %s", len(verdict_map), path
    )
    return verdict_map


class BooleanVerdictSignature(dspy.Signature):
    """DSPy signature that yields a boolean verdict for the claim."""

    claim_text = dspy.InputField(desc="Claim requiring verification.")
    retrieved_context = dspy.InputField(desc="Evidence snippets retrieved for the claim.")
    verdict = dspy.OutputField(
        desc="Return only the single word `true` if the claim is fully supported, or `false` otherwise.",
        prefix="",
    )


class BooleanVerdictProgram(dspy.Module):
    """Minimal DSPy program that outputs a boolean verdict without rationale."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(BooleanVerdictSignature)

    def forward(self, claim_text: str, retrieved_context: str) -> dspy.Response:
        return self.predictor(
            claim_text=claim_text,
            retrieved_context=retrieved_context,
        )


class CoTVerdictSignature(dspy.Signature):
    """Signature encouraging brief reasoning before the boolean verdict."""

    claim_text = dspy.InputField(desc="Claim requiring verification.")
    retrieved_context = dspy.InputField(desc="Relevant evidence snippets.")
    reasoning = dspy.OutputField(
        desc="Short chain-of-thought describing the verification steps.",
    )
    verdict = dspy.OutputField(
        desc="Final answer `true` or `false` only.",
        prefix="Answer:",
    )


class CoTVerdictProgram(dspy.Module):
    """Program that asks the model to think briefly before answering."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.ChainOfThought(CoTVerdictSignature)

    def forward(self, claim_text: str, retrieved_context: str) -> dspy.Response:
        return self.predictor(
            claim_text=claim_text,
            retrieved_context=retrieved_context,
        )


def boolean_verdict_metric(
    example: dspy.Example,
    prediction: dspy.Response,
    _trace: Any | None = None,
) -> float:
    """Heuristic metric rewarding clean boolean verdicts."""

    output = getattr(prediction, "verdict", "") or ""
    normalized = normalize_verdict_text(output)

    if not normalized:
        return 0.0

    if normalized == output.strip().lower():
        return 1.0

    return 0.7


def label_accuracy_metric(
    example: dspy.Example,
    prediction: dspy.Response,
    _trace: Any | None = None,
) -> float:
    """Metric that prioritizes accuracy against ground-truth labels."""

    gold = getattr(example, "processed_verdict", None)
    if not gold:
        return boolean_verdict_metric(example, prediction, _trace)

    predicted = normalize_verdict_text(getattr(prediction, "verdict", ""))
    if not predicted:
        return 0.0

    return 1.0 if predicted == gold else 0.0


def make_gepa_metric(
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float]
) -> Callable[
    [dspy.Example, dspy.Response, Any | None, str | None, Any | None],
    ScoreWithFeedback | float,
]:
    """Wrap our metric to provide GEPA-compatible feedback."""

    def gepa_metric(
        gold: dspy.Example,
        pred: dspy.Response,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> ScoreWithFeedback:
        score = metric_fn(gold, pred, trace)
        expected = getattr(gold, "processed_verdict", "unknown")
        predicted = normalize_verdict_text(getattr(pred, "verdict", ""))
        if predicted is None:
            feedback = "Prediction did not contain a parseable true/false verdict."
        else:
            feedback = f"Expected {expected}, predicted {predicted}."
        return ScoreWithFeedback(score=score, feedback=feedback)

    return gepa_metric


def optimize_program(
    program: BooleanVerdictProgram,
    train_examples: Sequence[dspy.Example],
    *,
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float],
    max_evals: int,
    gepa_mode: str,
    optimizer_lm: dspy.clients.LM | None,
    val_examples: Sequence[dspy.Example],
) -> BooleanVerdictProgram:
    """Run GEPA to optimize the boolean verdict program."""

    auto_arg: str | None = None if gepa_mode == "manual" else gepa_mode
    gepa_metric = make_gepa_metric(metric_fn)

    teleprompter = GEPA(
        metric=gepa_metric,
        auto=auto_arg,
        max_full_evals=max_evals if auto_arg is None else None,
        reflection_lm=optimizer_lm or dspy.settings.lm,
        track_stats=False,
    )

    logging.info(
        "Running GEPA optimization (mode=%s, max_full_evals=%s)",
        gepa_mode,
        max_evals if auto_arg is None else "auto",
    )

    optimized_program = teleprompter.compile(
        program,
        trainset=list(train_examples),
        valset=list(val_examples),
    )

    return optimized_program


def evaluate_program(
    program: dspy.Module,
    examples: Sequence[dspy.Example],
    *,
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float],
    verdict_attr: str,
    report_errors: bool,
    mode: str,
) -> DSPyArtifacts:
    """Evaluate the program on ``examples`` using the supplied metric."""

    scores: List[float] = []
    predictions: List[Dict[str, Any]] = []
    verdict_counts: Dict[str, int] = {"true": 0, "false": 0, "other": 0}

    for idx, example in enumerate(examples):
        prediction = program(
            claim_text=example.claim_text,
            retrieved_context=example.retrieved_context,
        )
        score = metric_fn(example, prediction, None)
        scores.append(score)

        raw_output = getattr(prediction, verdict_attr, "")
        normalized = normalize_verdict_text(raw_output)

        record: Dict[str, Any] = {
            "index": idx,
            "mode": mode,
            "domain": example.domain,
            "claim_text": example.claim_text,
            "raw_output": raw_output,
            "normalized_verdict": normalized,
            "score": score,
        }

        reasoning = getattr(prediction, "reasoning", None)
        if reasoning:
            record["reasoning"] = reasoning

        if normalized:
            verdict_counts[normalized] = verdict_counts.get(normalized, 0) + 1
        else:
            verdict_counts["other"] = verdict_counts.get("other", 0) + 1

        predictions.append(record)

        if report_errors and not normalized:
            logging.warning(
                "Could not parse boolean verdict for claim: %s",
                example.claim_text[:120],
            )

    average = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "examples": len(examples),
        "average_score": average,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
    }

    logging.info(
        "Evaluation results — examples=%d, avg=%.3f, min=%.3f, max=%.3f",
        summary["examples"],
        summary["average_score"],
        summary["min_score"],
        summary["max_score"],
    )

    summary["verdict_breakdown"] = {
        "true": verdict_counts.get("true", 0),
        "false": verdict_counts.get("false", 0),
        "other": verdict_counts.get("other", 0),
    }

    return DSPyArtifacts(program=program, metric_scores=summary, raw_predictions=predictions)


def compare_predictions_to_labels(
    predictions: Sequence[Dict[str, Any]],
    label_map: Dict[str, str],
) -> Dict[str, float | int]:
    """Compare normalized predictions to processed dataset verdicts."""

    total = 0
    correct = 0
    mismatched = 0
    missing_labels = 0
    unparsable = 0
    skipped_labels = 0

    for record in predictions:
        claim = record.get("claim_text", "")
        label_raw = label_map.get(claim)
        if label_raw is None:
            missing_labels += 1
            continue

        mapped_label = label_raw
        if mapped_label is None:
            skipped_labels += 1
            continue

        predicted = record.get("normalized_verdict", "")
        if not predicted:
            unparsable += 1
            continue

        total += 1
        if predicted == mapped_label:
            correct += 1
        else:
            mismatched += 1

    accuracy = (correct / total) if total else 0.0
    return {
        "matched_examples": total,
        "correct": correct,
        "mismatched": mismatched,
        "missing_labels": missing_labels,
        "skipped_labels": skipped_labels,
        "unparsable_predictions": unparsable,
        "accuracy": accuracy,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    rng = random.Random(args.seed)

    repo_root = args.repository_root

    claims: List[RawClaim] = []
    label_map: Dict[str, str] = {}
    try:
        processed_path = resolve_processed_file(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        sys.exit(1)

    claims, label_map = load_processed_claims(processed_path, limit=args.max_records)

    summarize_claims(claims)

    splits = split_train_test(claims, args.test_size, rng=rng)
    train_examples = convert_to_examples(splits.train, label_map)
    test_examples = convert_to_examples(splits.test, label_map)

    lm = initialize_lm(args.model_name)
    dspy.settings.configure(lm=lm)

    mode = args.mode

    has_labels = any(getattr(ex, "processed_verdict", None) for ex in train_examples + test_examples)
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float]

    optimizer_lm = initialize_optimizer_lm(args.optimizer_model)
    if optimizer_lm and mode != "dspy":
        logging.info(
            "Optimizer model specified but mode '%s' does not use DSPy optimization; ignoring.",
            mode,
        )
        optimizer_lm = None

    if mode == "simple":
        program: dspy.Module = BooleanVerdictProgram()
        metric_fn = label_accuracy_metric if has_labels else boolean_verdict_metric
        verdict_attr = "verdict"
        should_optimize = False
    elif mode == "cot":
        program = CoTVerdictProgram()
        metric_fn = label_accuracy_metric if has_labels else boolean_verdict_metric
        verdict_attr = "verdict"
        should_optimize = False
    else:
        program = CoTVerdictProgram()
        metric_fn = label_accuracy_metric if has_labels else boolean_verdict_metric
        verdict_attr = "verdict"
        should_optimize = args.optimize

    if should_optimize and train_examples:
        fit_examples, val_examples = split_fit_validation(
            train_examples,
            val_fraction=0.2,
            rng=rng,
        )
        program = optimize_program(
            program,
            fit_examples,
            metric_fn=metric_fn,
            max_evals=args.max_gepa_evals,
            gepa_mode=args.gepa_mode,
            optimizer_lm=optimizer_lm,
            val_examples=val_examples,
        )
    elif mode == "dspy" and not train_examples:
        logging.warning("No training examples available for DSPy optimization; running baseline.")
    elif mode == "dspy":
        logging.info("Skipping optimization — using baseline prompt only.")

    if mode in {"simple", "cot"} and args.optimize:
        logging.info("Optimization is only available in dspy mode; ignoring --optimize flag.")

    if args.preview and claims:
        preview_source = splits.test[0] if splits.test else splits.train[0]
        preview_text = build_prompt_preview_example(preview_source, mode)
        logging.info("Prompt preview:\n%s", preview_text)

    evaluation_examples = test_examples or train_examples
    artifacts = evaluate_program(
        program,
        evaluation_examples,
        metric_fn=metric_fn,
        verdict_attr=verdict_attr,
        report_errors=args.report_errors,
        mode=mode,
    )

    logging.info(
        "Final summary (%s mode): %s",
        mode,
        json.dumps(artifacts.metric_scores, indent=2),
    )

    if label_map:
        accuracy_report = compare_predictions_to_labels(artifacts.raw_predictions, label_map)
        logging.info(
            "Label comparison (%s mode): %s",
            mode,
            json.dumps(accuracy_report, indent=2),
        )


if __name__ == "__main__":
    main()
