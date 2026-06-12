# Repository Audit Before Cleanup
**Date:** June 11, 2026

This audit reviews the state of the `samatnext-v0.1` repository before any code refactoring and standardization are applied.

## 1. Current Configs
* `configs/samat_next_v0_1.json`: The core config for the SamatNext architecture (16 layers, hidden size 768, vocab size 151,936, with verifier head enabled).
* `configs/transformer_350m.json`: Config for the Transformer baseline.

## 2. Current Model Files
* `models/transformer_baseline.py`: PyTorch implementation of the decoder-only Transformer baseline (utilizes RoPE, SwiGLU, RMSNorm).
* `models/samat_next/config.py`: Configuration class `SamatNextConfig` for SamatNext.
* `models/samat_next/model.py`: Model wrappers `SamatNextModel` and `SamatNextForCausalLM`.
* `models/samat_next/layers.py`: Block structure `SamatNextBlock` alternating `GatedDeltaNet` and `DifferentialAttention`.
* `models/samat_next/deltanet.py`: Implementation of `GatedDeltaNet` as a simplified linear attention approximation.
* `models/samat_next/differential_attention.py`: Implementation of `DifferentialAttention` using scaled dot product attention.
* `models/samat_next/verifier.py`: Implementation of `VerifierHead` mapping the last token's hidden state to a logit.

## 3. Current Training Scripts
* `scripts/train_stage2a.py`: Stage 2A training (pretraining/syntax, hardcoded parameters, no argparser).
* `scripts/train_stage3.py`: Stage 3 training (semantics/paraphrase, has partial argparse for steps/resume).
* `scripts/train_stage5.py`: Stage 5 training (teacher distillation, hardcoded parameters).
* `scripts/train_transformer_curriculum.py`: Transformer curriculum training baseline (hardcoded parameters).

## 4. Current Evaluation Scripts
* `scripts/eval_humaneval.py`: Evaluates pass@1 on HumanEval (164 tasks).
* `scripts/eval_suite.py`: Custom evaluation suite running tests across Stages 2E, 3, 4B, 5, and HumanEval-5.
* `scripts/verify_config_truth.py`: Helper checking model checkpoint parameter sizes.
* `scripts/verify_repo_clean.py`: Cleanliness validation checker.

## 5. Current Data Files
* `data/HumanEval.jsonl` and `HumanEval.jsonl.gz`: Canonical HumanEval benchmark problems.
* `data/stage2a_code_pretrain.jsonl`: Code pretraining dataset.
* `data/stage3_paraphrase_train.jsonl` & `stage3_paraphrase_eval.jsonl`: Paraphrasing dataset.
* `data/stage4b_name_copy_train.jsonl` & `stage4b_name_copy_eval.jsonl`: Name binding/copy dataset.
* `data/stage5_teacher_distill.jsonl` & `stage5_teacher_distill_test.jsonl`: Teacher SFT data.
* `data/stage2e_adversarial_holdout.jsonl`: Adversarial evaluation dataset.
* Various other legacy stage datasets (e.g. stage2b, stage2c, stage2d).

## 6. Current Checkpoint Paths
Active/important checkpoints located in `checkpoints/`:
* `checkpoints/samat_next_350m_stage2a_fixed_best.pt`
* `checkpoints/samat_next_350m_stage3_best.pt`
* `checkpoints/samat_next_350m_stage5_best.pt`
* `checkpoints/transformer_350m_baseline_curriculum_stage5_best.pt`
* `checkpoints/transformer_350m_baseline_stage3_best.pt`
* `checkpoints/transformer_350m_baseline_stage5_best.pt`
* Various step and rescue checkpoints.

## 7. Current README Reproduction Commands
* Stage 2A: `python scripts/train_stage2a.py`
* Stage 3: `python scripts/train_stage3.py`
* Stage 5: `python scripts/train_stage5.py`
* Transformer Baseline: `python scripts/train_transformer_curriculum.py`
* Evaluation: `python scripts/compare_models.py`

## 8. Broken References and Missing Files
* **Missing Config:** Scripts `train_stage2a.py`, `train_stage3.py`, and `train_stage5.py` reference `configs/samat_next_150m.json`, which does not exist in the codebase.
* **Missing Script:** The README references `scripts/compare_models.py` as the evaluation command, which is completely missing from the codebase.

## 9. Result-Output Paths
* Stage 2A: `results/stage2a_log.json`
* Stage 3: `results/stage3_log.json`
* Stage 5: `results/stage5_log.json`
* Transformer Stage 5: `results/baseline_curriculum_stage5_log.json`

## 10. Script Compilation and Imports Status
All active scripts compile and import successfully (tested via py_compile and syntax parsing):
* `scripts/train_stage2a.py`: **PASS**
* `scripts/train_stage3.py`: **PASS**
* `scripts/train_stage5.py`: **PASS**
* `scripts/train_transformer_curriculum.py`: **PASS**
* `scripts/eval_humaneval.py`: **PASS**
* `scripts/eval_suite.py`: **PASS**
* `scripts/verify_config_truth.py`: **PASS**
* `scripts/verify_repo_clean.py`: **PASS**

## 11. Command Line Arguments vs Hardcoded Constants
* `scripts/train_stage2a.py`: Fully hardcoded constants.
* `scripts/train_stage3.py`: Mix of hardcoded constants and arguments for `--steps`, `--resume`, and `--start_step`.
* `scripts/train_stage5.py`: Fully hardcoded constants.
* `scripts/train_transformer_curriculum.py`: Fully hardcoded constants.
* `scripts/eval_humaneval.py`: Fully hardcoded constants.
* `scripts/eval_suite.py`: Fully hardcoded constants.

## 12. Mismatches Between README and Codebase
* README references `compare_models.py` which is missing.
* README mentions "matched Transformer baseline," but `configs/transformer_350m.json` has `vocab_size: 151665` while SamatNext has `vocab_size: 151936`.
* Parameter discrepancy:
  - SamatNext model is ~356M parameters.
  - Transformer baseline is ~346M parameters (mismatch of ~9.4M parameters due to extra projections in DifferentialAttention).
* Config fields like `attention_ratio`, `deltanet_ratio`, and `use_differential_attention` are exposed in `SamatNextConfig` but ignored inside `layers.py` where alternation is hardcoded using `layer_idx % 2 == 0`.
* The `verifier_head` is defined in configs and instantiated, but is completely unused during training.
