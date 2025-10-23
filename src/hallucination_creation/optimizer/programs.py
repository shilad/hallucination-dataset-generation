from __future__ import annotations

from typing import Any, Callable, Tuple

import dspy


class BooleanVerdictSignature(dspy.Signature):
    claim_text = dspy.InputField(desc="Claim requiring verification.")
    retrieved_context = dspy.InputField(desc="Evidence snippets retrieved for the claim.")
    verdict = dspy.OutputField(
        desc="Return only the single word `true` if the claim is fully supported, or `false` otherwise.",
        prefix="",
    )


class BooleanVerdictProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(BooleanVerdictSignature)

    def forward(self, claim_text: str, retrieved_context: str) -> dspy.Response:
        return self.predictor(claim_text=claim_text, retrieved_context=retrieved_context)


class CoTVerdictSignature(dspy.Signature):
    claim_text = dspy.InputField(desc="Claim requiring verification.")
    retrieved_context = dspy.InputField(desc="Relevant evidence snippets.")
    reasoning = dspy.OutputField(desc="Short chain-of-thought describing the verification steps.")
    verdict = dspy.OutputField(desc="Final answer `true` or `false` only.", prefix="Answer:")


class CoTVerdictProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.ChainOfThought(CoTVerdictSignature)

    def forward(self, claim_text: str, retrieved_context: str) -> dspy.Response:
        return self.predictor(claim_text=claim_text, retrieved_context=retrieved_context)


class CoTRationaleSignature(dspy.Signature):
    claim_text = dspy.InputField(desc="Claim requiring verification.")
    retrieved_context = dspy.InputField(desc="Relevant evidence snippets.")
    rationale = dspy.OutputField(desc="Short factual rationale grounded in the evidence.", prefix="Reasoning:")
    verdict = dspy.OutputField(desc="Final answer `true` or `false` only.", prefix="Answer:")


class CoTRationaleProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.ChainOfThought(CoTRationaleSignature)

    def forward(self, claim_text: str, retrieved_context: str) -> dspy.Response:
        return self.predictor(claim_text=claim_text, retrieved_context=retrieved_context)


def choose_program_and_metric(
    *,
    mode: str,
    has_labels: bool,
    optimize_flag: bool,
    use_rationale: bool = False,
    verdict_weight: float = 0.8,
) -> Tuple[dspy.Module, Callable[[dspy.Example, dspy.Response, Any | None], float], str, bool]:
    from .metrics import (
        boolean_verdict_metric,
        combine_verdict_and_rationale_metrics,
        label_accuracy_metric,
        rationale_quality_metric,
    )

    base_metric: Callable[[dspy.Example, dspy.Response, Any | None], float]
    base_metric = label_accuracy_metric if has_labels else boolean_verdict_metric
    metric_fn = (
        combine_verdict_and_rationale_metrics(base_metric, rationale_quality_metric, verdict_weight=verdict_weight)
        if use_rationale
        else base_metric
    )

    if mode == "simple":
        if use_rationale:
            raise ValueError("Rationale mode is unsupported for 'simple' programs.")
        return BooleanVerdictProgram(), metric_fn, "verdict", False
    if mode == "cot":
        program: dspy.Module = CoTRationaleProgram() if use_rationale else CoTVerdictProgram()
        return program, metric_fn, "verdict", False
    if mode == "dspy":
        program = CoTRationaleProgram() if use_rationale else CoTVerdictProgram()
        return program, metric_fn, "verdict", optimize_flag
    raise ValueError(f"Unsupported mode '{mode}'. Choose from simple, cot, or dspy.")
