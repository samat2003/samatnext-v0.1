# SamatNext-v0.1 Data Pipeline

This directory contains the raw data sources, processed splits, and benchmarks for the SamatNext curriculum evaluation.

## Dataset Structure
- `raw/`: Uncleaned raw stage datasets.
- `processed/`: Cleaned, filtered, decontaminated, and split datasets.
- `benchmark/`: Test datasets for HumanEval, MBPP, and custom retention tasks.
- `manifests/`: Manifest metadata and SHA256 hashes list.

## Regeneration Instructions
To regenerate the processed and benchmark datasets, run the following command from the repository root:

```bash
# Standard run
python scripts/prepare_data.py

# Smoke test run (generates a fast, tiny subset)
python scripts/prepare_data.py --smoke
```

Alternatively, you can use the Makefile targets:
```bash
make prepare-data
make prepare-data-smoke
```

## Data Cleaning and Decontamination
The data preparation script automatically performs:
1. Malformed JSONL line removal.
2. Empty prompt/target filtering.
3. Markdown code block stripping.
4. Exact and near-duplicate removal using content hashes.
5. Strict AST parsing to verify Python syntax validity.
6. Contamination filtering against HumanEval and MBPP test tasks.
