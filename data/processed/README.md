## Processed Hallucination Dataset

Curated datasets produced by the scripted pipeline live here. Each artifact is
stored as `dataset_YYYY-MM-DD_HHMMSS.parquet` (or `.csv` when Parquet is not
available) and contains the standardized schema:

- `claim_text`: Statement produced by the generator.
- `verdict`: `supported`, `unsupported`, or `mixed`.
- `explanation`: Natural-language justification referencing retrieved evidence.
- `domain`: Source domain for the claim.
- `collected_at`: Timestamp inherited from the raw record.
- `split`: `train` or `test` flag based on a 60/40 chronological split.
- `reference_1` / `reference_2` / `reference_3`: Top evidence sources (title and URL) supporting the labeling decision.
- `snippet_1` / `snippet_2` / `snippet_3`: Corresponding evidence excerpts.

Whenever you adjust the schema or labeling policy, add a dated note below:

- 2025-10-13: Initialized processed dataset schema for scripted pipeline.
- 2025-10-13: Added `split` column using deterministic 60/40 assignment.
- 2025-10-13: Constrained claim generation and evidence gathering to information available through 2023.
- 2025-10-13: Updated claim generation instructions to emphasize multi-step reasoning scenarios.
- 2025-10-13: Pipeline now discards evaluations marked as uncertain, keeping only clearly supported or unsupported claims.
- 2025-10-13: Export step can optionally trim surplus rows to keep supported and unsupported verdict counts balanced.
- 2025-10-13: Added top-three evidence reference and snippet columns to processed exports.
