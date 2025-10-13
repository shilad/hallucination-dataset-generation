# Repository Guidelines

## Project Structure & Module Organization
Development now happens in Python modules under `src/hallucination_creation/`.
This repository is an instructional example for my class, so keep the structure
approachable and well-commented. `main.py` boots an agent that coordinates
prompt construction, web retrieval, and labeling, storing untouched model
responses in `data/raw/` and curated datasets in `data/processed/`. Keep both
directories accompanied by `README` files that summarize schema revisions and
date-stamped exports. `.venv/` is disposable, and editor metadata such as
`HallucinationCreation.iml` should remain out of version control unless
intentionally shared.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create an isolated interpreter aligned with collaborators.
- `pip install -e .` or `uv sync`: install project dependencies.
- `uv run python main.py`: execute the web-enabled dataset agent end to end.
- `pytest`: validate shared utilities and regression tests.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indents, snake_case for helpers, UpperCamelCase for
classes. Organize modules by responsibility (e.g., `web_tools.py`,
`prompt_builder.py`, `labeling.py`) and mirror that structure in `tests/`.
Document major agent flows with module-level docstrings or concise comments to
keep reasoning steps readable in diffs.

## Testing Guidelines
Use `pytest` to cover prompt builders, response parsers, retrieval fallbacks,
and dataset validators. Seed randomness (`random.seed(42)`) before generating
sample responses so tests stay deterministic. Include a smoke test that loads
processed datasets and asserts required columns before any export.

## Commit & Pull Request Guidelines
Write commits using `type: summary`, such as `feat: add web-backed retrieval`.
Detail dataset impacts in the body (row deltas, new labels) and reference the
agent modules involved. Pull requests should summarize experimental intent,
link tracking issues, and include tables or snapshots describing dataset
changes or evaluation metrics.

## Dataset Management & Security
Keep API keys and model endpoints in environment variables loaded via `.env`
(add a `.env.example`). Log prompts and responses without sensitive tokens;
scrub PII before committing. Version large dataset exports with date-stamped
filenames (e.g., `processed_2024-05-01.parquet`) and prefer publishing large
artifacts to managed storage rather than Git when policy limits apply.
