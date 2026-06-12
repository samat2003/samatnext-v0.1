# Dataset License & Redistribution Manifest

Every dataset used in the curriculum learning setup is documented below. In accordance with open-science reproducibility standards, no third-party datasets are committed directly to the repository unless their licenses explicitly permit redistribution. Otherwise, data builder scripts, manifests, and hashes are committed, and users must download/regenerate them locally.

## 1. HumanEval
- **Dataset ID:** `openai_humaneval` / `HumanEval.jsonl.gz`
- **License:** MIT License
- **Redistribution:** Allowed. A local pointer (`data/HumanEval.jsonl`) is maintained for offline execution tests.
- **Contamination Filtering:** Completely held out. Never used in any training curriculum stage.

## 2. MBPP (Mostly Basic Python Problems)
- **Dataset ID:** `google-research-datasets/mbpp`
- **License:** CC-BY-4.0
- **Redistribution:** Allowed.
- **Redistribution Policy:** Only held-out evaluation datasets or metadata pointers are generated. Never used in training.

## 3. Flytech Python Codes (Flytech/python-codes-25k)
- **Source:** Hugging Face `Flytech/python-codes-25k`
- **License:** Apache License 2.0
- **Redistribution:** Allowed.
- **Redistribution Policy:** Manifests and pre-filtering hash files are checked in; the raw files are rebuilt using `prepare_data.py`.

## 4. CodeFeedback
- **Source:** Hugging Face `m-a-p/CodeFeedback-Filtered-Instruction`
- **License:** MIT License
- **Redistribution:** Allowed.

## 5. Teacher-Generated SFT Examples (Stage 5)
- **Source:** Outputs synthetically generated from Qwen2.5-Coder-3B-Instruct.
- **License:** Apache License 2.0 (under Qwen2.5 usage conditions).
- **Redistribution:** Allowed. Committed under `data/stage5_teacher_distill.jsonl` for exact baseline comparison.
