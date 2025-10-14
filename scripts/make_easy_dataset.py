#!/usr/bin/env python3
"""Generate an easier variant of a processed hallucination dataset.

The script reads a processed CSV in ``data/processed`` and produces a new CSV where
``claim_text`` (and optionally the explanatory fields) are rewritten to be
substantially easier for students to interpret. The rewrite is delegated to
OpenAI's ``gpt-5`` reasoning model with medium effort so we preserve fidelity
while simplifying wording and structure.

Environment:
    - ``OPENAI_API_KEY`` must be set in your shell (or a loaded ``.env`` file).

Example usage::

    uv run python scripts/make_easy_dataset.py \
        --input data/processed/dataset_2025-10-13_222800.csv \
        --output data/processed/dataset_2025-10-13_222800_easy.csv

The script deliberately avoids in-place edits; the original dataset remains
untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenAI's official client is required. Install with `pip install openai`."
    ) from exc


SYSTEM_PROMPT = """You redesign factual verification tasks so they stay rigorous yet more approachable.
Goals:
1. Retain the original claim's truth value and essential evidence. The verdict must remain valid.
2. Simplify both wording *and* investigative scope so the task is easier than the original, yet still requires careful thought—challenging for a strong high-school graduate but not trivial.
3. You may narrow the claim, remove distracting subpoints, or add context that clarifies what to check, as long as the answer is unambiguous.
4. Keep the dataset structure: return rewritten `claim_text`, and—if helpful—adjust `explanation` and up to three snippets to match the easier task.
5. Output strictly as JSON with keys: claim_text, explanation, snippet_1, snippet_2, snippet_3.
"""

USER_TEMPLATE = """Original claim (verdict: {verdict}):\n{claim_text}\n\nSnippets:\n1) {snippet1}\n2) {snippet2}\n3) {snippet3}\n\nExplanation:\n{explanation}\n\nRewrite the claim so it is substantially easier to read.\nReturn JSON only."""


@dataclass
class Row:
    claim_text: str
    verdict: str
    explanation: str
    snippet_1: str
    snippet_2: str
    snippet_3: str
    original: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the processed CSV you want to simplify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path for the easier dataset.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model identifier to use for rewriting (default: gpt-5).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process (debugging helper).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel rewrite workers (default: 5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_rows(path: Path, limit: Optional[int]) -> List[Row]:
    logging.info("Loading dataset: %s", path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"claim_text", "verdict", "explanation", "snippet_1", "snippet_2", "snippet_3"}
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"Input CSV missing required columns: {', '.join(sorted(missing))}")
        rows: List[Row] = []
        for idx, raw in enumerate(reader):
            row = Row(
                claim_text=raw.get("claim_text", ""),
                verdict=raw.get("verdict", ""),
                explanation=raw.get("explanation", ""),
                snippet_1=raw.get("snippet_1", ""),
                snippet_2=raw.get("snippet_2", ""),
                snippet_3=raw.get("snippet_3", ""),
                original=raw,
            )
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    logging.info("Loaded %d row(s)", len(rows))
    return rows


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def build_user_prompt(row: Row) -> str:
    return USER_TEMPLATE.format(
        verdict=row.verdict or "unknown",
        claim_text=row.claim_text or "<empty>",
        snippet1=row.snippet_1 or "<empty>",
        snippet2=row.snippet_2 or "<empty>",
        snippet3=row.snippet_3 or "<empty>",
        explanation=row.explanation or "<empty>",
    )


def call_model(client: OpenAI, model: str, user_prompt: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    if hasattr(response, "output_text"):
        message = response.output_text
    else:
        chunks = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    chunks.append(getattr(part, "text", ""))
        message = "".join(chunks)

    logging.debug("Model raw output: %s", message)
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        start = message.find("{")
        end = message.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(message[start : end + 1])


def merge_row(row: Row, rewrite: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(row.original)
    merged["claim_text"] = rewrite.get("claim_text", row.claim_text)
    merged["explanation"] = rewrite.get("explanation", row.explanation)
    merged["snippet_1"] = rewrite.get("snippet_1", row.snippet_1)
    merged["snippet_2"] = rewrite.get("snippet_2", row.snippet_2)
    merged["snippet_3"] = rewrite.get("snippet_3", row.snippet_3)
    merged.setdefault("notes", "")
    merged["notes"] = (merged["notes"] + "\nEasy rewrite generated").strip()
    return merged


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)  # type: ignore[arg-type]
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info("Wrote %d row(s) to %s", len(rows), path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY is not set; export it or load a .env file.")
        sys.exit(1)

    ensure_output_path(args.output, args.overwrite)
    rows = load_rows(args.input, args.max_rows)

    client = OpenAI(api_key=api_key)

    def process(idx_row: Tuple[int, Row]) -> Tuple[int, Dict[str, str]]:
        idx, row = idx_row
        prompt = build_user_prompt(row)
        local_client = OpenAI(api_key=api_key)
        payload = call_model(local_client, args.model, prompt)
        merged = merge_row(row, payload)
        return idx, merged

    rewritten: List[Optional[Dict[str, str]]] = [None] * len(rows)
    total = len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(process, (idx, row)): idx
            for idx, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            idx = future_map[future]
            try:
                original_idx, merged = future.result()
            except Exception as exc:  # pragma: no cover - network path
                logging.error("Rewrite failed on row %d: %s", idx + 1, exc)
                executor.shutdown(cancel_futures=True)
                sys.exit(1)
            rewritten[original_idx] = merged
            logging.info("Progress: %d/%d rows complete", completed, total)

    filtered_rows = [row for row in rewritten if row]
    # Preserve original column ordering while allowing new metadata columns such as "notes".
    if rows:
        base_fieldnames = list(rows[0].original.keys())
        merged_fieldnames = list(base_fieldnames)
        for entry in filtered_rows:
            for key in entry.keys():
                if key not in merged_fieldnames:
                    merged_fieldnames.append(key)
    else:
        merged_fieldnames = []

    write_rows(args.output, merged_fieldnames, filtered_rows)


if __name__ == "__main__":
    main()
