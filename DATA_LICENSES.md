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
- **License:** MIT
- **Redistribution:** Allowed.
- **Redistribution Policy:** Manifests and pre-filtering hash files are checked in; the raw files are rebuilt using `prepare_data.py`.

## 4. CodeFeedback
- **Source:** Hugging Face `m-a-p/CodeFeedback-Filtered-Instruction`
- **License:** Apache-2.0
- **Redistribution:** Allowed.

## 5. Teacher-Generated SFT Examples (Stage 5)
- **Source:** Synthetic outputs generated from Qwen2.5-Coder-3B-Instruct.
- **License:** License-sensitive; subject to upstream Qwen Research license terms.
- **Redistribution:** Not automatically allowed by this repository’s Apache-2.0 source-code license.
- **Repository policy:** Do not commit the full teacher-generated dataset unless redistribution rights are verified. Prefer builder scripts, manifests, and hashes.
- **Current public repository status:** Full Stage 5 dataset is not committed; public `data/` contains only preview/metadata unless otherwise stated.
- **Licensing Notice:** Teacher-generated SFT examples were generated using Qwen2.5-Coder-3B-Instruct. The upstream Qwen2.5-Coder-3B model is subject to the Qwen Research license, not Apache-2.0. Redistribution and downstream use of generated examples should be treated conservatively and verified against upstream terms before public dataset release. These teacher-generated examples are license-sensitive and are not automatically covered by the repository’s Apache-2.0 source-code license.

