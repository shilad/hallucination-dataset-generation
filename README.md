## Hallucination Dataset Generation

This repository now manages the hallucination-detection dataset entirely in
code. It serves as a worked example for my class on advanced LLM operations,
showing how to scaffold a web-grounded hallucination detection pipeline.
The primary entry point is a Python agent that orchestrates prompt generation,
web-grounded fact gathering, and labeling to decide whether model claims are
supported by retrieved evidence.

### Setup

```bash
uv sync
```

If you prefer `pip`, install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Usage

Run the orchestration script to execute the dataset agent:

```bash
uv run python main.py --max-parallel-llm 3
```

Add `--balance-verdicts` if you want to trim the processed dataset to equal counts of supported and unsupported claims before export.
Use `--num-domains 1` to restrict the run to a single domain during smoke testing.

The script wires up the web-enabled agent components from
`src/hallucination_creation/` and exports intermediate artifacts to
`data/raw/`, with curated datasets written to `data/processed/`. See the
README files in those folders for schema notes and change history.

You can also call the collection entry point directly:

```bash
uv run python scripts/collect_dataset.py --max-claims-per-domain 3 --max-parallel-llm 3
```

Append `--balance-verdicts` to enforce verdict balancing in this entry point.
Append `--num-domains` to limit the number of domains processed.

Evaluate the latest processed dataset with the lightweight GPT-4.1-mini baseline:

```bash
uv run python scripts/evaluate_dataset.py --split test --max-parallel-llm 3
```

The pipeline spans several components:
- `src/hallucination_creation/domains.py`: curated domain prompts to encourage diverse claims.
- `src/hallucination_creation/claim_generator.py`: reasoning model prompts for candidate claims (limited to information available through 2023 and biased toward multi-step reasoning) running under semaphore-managed async generation.
- `src/hallucination_creation/retriever.py`: web-enabled evidence gathering (OpenAI Responses API with web search, restricted to sources published on or before 2023-12-31) with semaphore-controlled concurrency.
- `src/hallucination_creation/evaluator.py`: hallucination assessor that produces `{claim_text, verdict, explanation}` and labels records as clear/ambiguous so the pipeline can drop uncertain cases.
- `src/hallucination_creation/simple_evaluator.py`: baseline GPT-4.1-mini classifier used by the evaluation script.
- `src/hallucination_creation/writer.py`: incremental writers for raw JSONL and processed Parquet/CSV outputs (adds a 60/40 `train`/`test` split, captures top-three evidence references/snippets, optionally balances supported/unsupported counts, and is driven by an async writer lock).
- `src/hallucination_creation/pipeline.py`: orchestrates generation → retrieval → evaluation → persistence.

### Environment variables

Duplicate `.env.example` into `.env` and provide your actual credentials:

```bash
cp .env.example .env
```

Populate `OPENAI_API_KEY` and any vendor-specific settings required by your web
tools. Optional variables:
- `OPENAI_MODEL`: model powering hallucination evaluation (default `gpt-5`, using medium reasoning effort).
- `OPENAI_REASONING_EFFORT`: override the evaluator reasoning effort (`"medium"` by default).
- `OPENAI_RETRIEVER_MODEL`: model used by the web retriever (default `gpt-5` with medium reasoning effort).
- `OPENAI_RETRIEVER_REASONING_EFFORT`: override the retriever's reasoning effort if you need `"high"` or `"minimal"` instead of the default `"medium"`.

### Testing

Run the smoke test suite to verify the scripted pipeline with mocked components:

```bash
uv run pytest
```

Additional tests should target prompt builders, evidence parsing, and dataset validators as modules evolve.
