from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Set

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


def normalize_verdict_text(text: str | None) -> str | None:
    if not text:
        return None
    val = text.strip().lower()
    if val in {"true", "false"}:
        return val
    # Allow trivial formatting around the answer
    if val.endswith("true"):
        return "true"
    if val.endswith("false"):
        return "false"
    return None


def boolean_verdict_metric(example: dspy.Example, prediction: dspy.Response, _trace: Any | None = None) -> float:
    output = getattr(prediction, "verdict", "") or ""
    normalized = normalize_verdict_text(output)
    if not normalized:
        return 0.0
    return 1.0 if normalized == output.strip().lower() else 0.7


def label_accuracy_metric(example: dspy.Example, prediction: dspy.Response, _trace: Any | None = None) -> float:
    gold = getattr(example, "processed_verdict", None)
    if not gold:
        return boolean_verdict_metric(example, prediction, _trace)
    predicted = normalize_verdict_text(getattr(prediction, "verdict", ""))
    if not predicted:
        return 0.0
    return 1.0 if predicted == gold else 0.0


def _extract_keywords(text: str) -> Set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]{4,}", text or "")}


def rationale_quality_metric(example: dspy.Example, prediction: dspy.Response, _trace: Any | None = None) -> float:
    rationale = getattr(prediction, "rationale", "") or getattr(prediction, "reasoning", "")
    rationale = rationale.strip()
    if not rationale:
        return 0.0

    score = 0.4  # base credit for providing an explanation
    lower = rationale.lower()
    if lower.startswith("clear:") or lower.startswith("uncertain:"):
        score += 0.2

    predicted = normalize_verdict_text(getattr(prediction, "verdict", ""))
    gold = getattr(example, "processed_verdict", None)
    if lower.startswith("clear:") and gold:
        score += 0.2 if predicted == gold else -0.2

    rationale_terms = _extract_keywords(rationale)
    context_terms = _extract_keywords(getattr(example, "retrieved_context", ""))
    overlap = len(rationale_terms & context_terms)
    if overlap >= 3:
        score += 0.2
    elif overlap >= 1:
        score += 0.1
    else:
        score -= 0.1

    gold_explanation = getattr(example, "explanation", "")
    if gold_explanation:
        gold_terms = _extract_keywords(gold_explanation)
        gold_overlap = len(rationale_terms & gold_terms)
        if gold_overlap >= 3:
            score += 0.1
        elif gold_overlap == 0:
            score -= 0.1

    return max(0.0, min(1.0, score))


def combine_verdict_and_rationale_metrics(
    verdict_metric: Callable[[dspy.Example, dspy.Response, Any | None], float],
    rationale_metric: Callable[[dspy.Example, dspy.Response, Any | None], float],
    *,
    verdict_weight: float = 0.8,
) -> Callable[[dspy.Example, dspy.Response, Any | None], float]:
    rationale_weight = 1.0 - verdict_weight

    def combined_metric(example: dspy.Example, prediction: dspy.Response, trace: Any | None = None) -> float:
        verdict_score = verdict_metric(example, prediction, trace)
        rationale_score = rationale_metric(example, prediction, trace)
        return (verdict_weight * verdict_score) + (rationale_weight * rationale_score)

    return combined_metric


def make_gepa_metric(
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float]
) -> Callable[[dspy.Example, dspy.Response, Any | None, str | None, Any | None], ScoreWithFeedback]:
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
        rationale_present = bool((getattr(pred, "rationale", "") or getattr(pred, "reasoning", "")).strip())
        feedback = (
            "Prediction did not contain a parseable true/false verdict."
            if predicted is None
            else f"Expected {expected}, predicted {predicted}. "
            + ("Rationale missing." if not rationale_present else "")
        )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return gepa_metric


def evaluate_program(
    program: dspy.Module,
    examples: Sequence[dspy.Example],
    *,
    metric_fn: Callable[[dspy.Example, dspy.Response, Any | None], float],
    verdict_attr: str,
    report_errors: bool,
    mode: str,
):
    scores: List[float] = []
    predictions: List[Dict[str, Any]] = []
    verdict_counts: Dict[str, int] = {"true": 0, "false": 0, "other": 0}

    for idx, example in enumerate(examples):
        prediction = program(claim_text=example.claim_text, retrieved_context=example.retrieved_context)
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
        rationale = getattr(prediction, "rationale", None)
        if rationale:
            record["rationale"] = rationale
        if normalized:
            verdict_counts[normalized] = verdict_counts.get(normalized, 0) + 1
        else:
            verdict_counts["other"] = verdict_counts.get("other", 0) + 1
        predictions.append(record)
        if report_errors and not normalized:
            import logging

            logging.warning("Could not parse boolean verdict for claim: %s", example.claim_text[:120])

    average = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "examples": len(examples),
        "average_score": average,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "verdict_breakdown": {
            "true": verdict_counts.get("true", 0),
            "false": verdict_counts.get("false", 0),
            "other": verdict_counts.get("other", 0),
        },
    }
    return type("DSPyArtifacts", (), {"program": program, "metric_scores": summary, "raw_predictions": predictions})()


def compare_predictions_to_labels(predictions: Sequence[Dict[str, Any]], label_map: Dict[str, str]) -> Dict[str, float | int]:
    total = correct = mismatched = missing_labels = unparsable = skipped_labels = 0
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
