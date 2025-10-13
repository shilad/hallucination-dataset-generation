## Processed Hallucination Dataset

Curated datasets produced by the scripted pipeline live here. Each artifact is
stored as `dataset_YYYY-MM-DD_HHMMSS.parquet` (or `.csv` when Parquet is not
available) and contains the standardized schema:

- `claim_text`: Statement produced by the generator.
- `verdict`: `supported`, `unsupported`, or `mixed`.
- `explanation`: Natural-language justification referencing retrieved evidence.
- `domain`: Source domain for the claim.
- `collected_at`: Timestamp inherited from the raw record.

Whenever you adjust the schema or labeling policy, add a dated note below:

- 2025-10-13: Initialized processed dataset schema for scripted pipeline.
