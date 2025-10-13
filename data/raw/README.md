## Raw Hallucination Samples

This directory stores untouched model interactions captured by the scripted
collection pipeline. Each JSON Lines file is named
`claims_YYYY-MM-DD_HHMMSS.jsonl` and contains the following fields per row:

- `domain`: Domain prompt or scenario used when generating the claim.
- `prompt`: Direct prompt sent to the reasoning model to create the claim.
- `claim_text`: Generated claim text prior to any filtering.
- `generator_reasoning`: Optional reasoning returned by the claim generator.
- `retrieved_context`: Evidence snippets captured by web tools, when present.
- `collected_at`: ISO 8601 timestamp for traceability.

Raw files should remain immutable so downstream scripts can reproduce label
decisions. Document schema changes here when fields are added or removed.
