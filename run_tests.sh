#!/bin/bash

#python scripts/dspy_prompt_optimizer.py --mode simple --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42
python scripts/dspy_prompt_optimizer.py --mode cot --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42
#python scripts/dspy_prompt_optimizer.py --mode cot --use-rationale --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42
#python scripts/dspy_prompt_optimizer.py --mode dspy --optimize=false --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42
#python scripts/dspy_prompt_optimizer.py --mode dspy --use-rationale --optimize=false --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42
#python scripts/dspy_prompt_optimizer.py --mode dspy --use-rationale --processed-file data/processed/dataset_2025-10-13_222800_easy.csv --seed 42