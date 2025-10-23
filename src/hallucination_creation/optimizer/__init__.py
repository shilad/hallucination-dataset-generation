"""Student-friendly DSPy optimizer building blocks.

This package splits the original monolithic script into small, readable pieces
without changing behavior. The CLI in `scripts/dspy_prompt_optimizer.py` uses
these helpers to keep top-level flow clear.
"""

from .constants import DEFAULT_MODEL, DEFAULT_OPTIMIZER_MODEL
from .io import (
    RawClaim,
    DatasetSplits,
    resolve_processed_file,
    load_processed_claims,
    split_train_test,
    convert_to_examples,
    summarize_claims,
)
from .optimize import initialize_lm, initialize_optimizer_lm, optimize_program
from .programs import (
    BooleanVerdictProgram,
    CoTVerdictProgram,
    CoTRationaleProgram,
    choose_program_and_metric,
)
from .metrics import (
    label_accuracy_metric,
    boolean_verdict_metric,
    rationale_quality_metric,
    combine_verdict_and_rationale_metrics,
    evaluate_program,
    compare_predictions_to_labels,
)

__all__ = [
    # constants
    "DEFAULT_MODEL",
    "DEFAULT_OPTIMIZER_MODEL",
    # io/types
    "RawClaim",
    "DatasetSplits",
    "resolve_processed_file",
    "load_processed_claims",
    "split_train_test",
    "convert_to_examples",
    "summarize_claims",
    # programs
    "BooleanVerdictProgram",
    "CoTVerdictProgram",
    "CoTRationaleProgram",
    "choose_program_and_metric",
    # metrics
    "label_accuracy_metric",
    "boolean_verdict_metric",
    "rationale_quality_metric",
    "combine_verdict_and_rationale_metrics",
    "evaluate_program",
    "compare_predictions_to_labels",
    # optimize
    "initialize_lm",
    "initialize_optimizer_lm",
    "optimize_program",
]
