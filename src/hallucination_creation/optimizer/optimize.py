from __future__ import annotations

import logging
import re
from typing import Any, Dict

import dspy


def reasoning_lm_kwargs(model_identifier: str, effort: str | None = None) -> Dict[str, Any]:
    base = model_identifier.split("/")[-1].lower()
    if re.match(r"^(?:o[1345]|gpt-5)(?:-[a-z]+)?", base):
        kwargs: Dict[str, Any] = {"temperature": 1.0, "max_tokens": 16000}
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs
    return {}


def initialize_lm(model_name: str) -> dspy.clients.LM:
    logging.debug("Initializing DSPy LM wrapper with model '%s'", model_name)
    provider_model = model_name if "/" in model_name else f"openai/{model_name}"
    kwargs = reasoning_lm_kwargs(provider_model)
    return dspy.clients.LM(model=provider_model, model_type="chat", **kwargs)


def initialize_optimizer_lm(model_name: str | None) -> dspy.clients.LM | None:
    if not model_name:
        return None
    logging.debug("Initializing optimizer LM with model '%s'", model_name)
    provider_model = model_name if "/" in model_name else f"openai/{model_name}"
    kwargs = reasoning_lm_kwargs(provider_model, effort="low")
    return dspy.clients.LM(model=provider_model, model_type="chat", **kwargs)


def optimize_program(
    program: dspy.Module,
    train_examples,
    *,
    metric,
    max_evals: int,
    gepa_mode: str,
    optimizer_lm,
    val_examples,
):
    from dspy.teleprompt import GEPA
    from .metrics import make_gepa_metric

    auto_arg = None if gepa_mode == "manual" else gepa_mode
    gepa_metric = make_gepa_metric(metric)
    teleprompter = GEPA(
        metric=gepa_metric,
        auto=auto_arg,
        max_full_evals=max_evals if auto_arg is None else None,
        reflection_lm=optimizer_lm or dspy.settings.lm,
        track_stats=False,
    )
    logging.info("Running GEPA optimization (mode=%s, max_full_evals=%s)", gepa_mode, max_evals if auto_arg is None else "auto")
    return teleprompter.compile(program, trainset=list(train_examples), valset=list(val_examples))

