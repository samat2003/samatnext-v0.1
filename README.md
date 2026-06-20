# SamatNext-v0.1

[![CI](https://github.com/samat2003/samatnext-v0.1/actions/workflows/ci.yml/badge.svg)](https://github.com/samat2003/samatnext-v0.1/actions/workflows/ci.yml)

## Summary
SamatNext-v0.1 is an experimental ~350M-parameter autoregressive language model architecture designed to study retention under staged curriculum training for Python code tasks. The project compares a hybrid decoder using Differential-Attention-style layers and DeltaNet-inspired simplified linear-state mixers against a parameter-matched Transformer baseline.

## Research Claim
SamatNext-v0.1 shows substantially stronger intermediate semantic curriculum retention than a parameter-matched Transformer sequential fine-tuning baseline under a controlled Python curriculum.

This should be interpreted as evidence of an altered retention/plasticity tradeoff, not as evidence that the architecture is a solution to catastrophic forgetting in general.

## Scope of Claim
This project does **not** claim that SamatNext-v0.1 is a general replacement for Transformers, a SOTA code model, or a complete solution to catastrophic forgetting.

The supported claim is narrower: in this controlled staged Python curriculum, SamatNext-v0.1 preserves much more Stage 3 semantic behavior after final-stage training than the strongest parameter-matched Transformer sequential fine-tuning baseline tested here. Long-horizon early-stage syntax retention remains weak for both architectures.

## What This Is / Is Not
### What This Is:
- An experimental research prototype (~350M parameters).
- A controlled curriculum-retention study for small code models.
- A reproducibility-oriented artifact with parameter-matched baselines, Makefile commands, CI checks, and archived evaluation outputs.

### What This Is NOT:
- **Not** a general SOTA frontier coding assistant.
- **Not** production-ready.
- **Not** a general replacement for standard Transformers across all NLP tasks.
- **Not** trained or evaluated on complete HumanEval for benchmarks.
- **Not** a claim that the catastrophic forgetting problem is solved.

## Architecture
The **SamatNext-v0.1** autoregressive next-token decoder integrates:
- **Differential-Attention-style layers:** Attention blocks inspired by differential attention mechanisms.
- **DeltaNet-inspired simplified linear-state mixer layers:** Recurrent/linear-state sequence-mixing blocks intended to test whether state-tracking inductive bias affects curriculum retention.
- **Feed-forward blocks:** Standard SwiGLU non-linear mappings.

## Matched Transformer Baseline
The Transformer baseline is parameter-matched to SamatNext within a strict 0.01% parameter-count tolerance. The main comparison should be read as a comparison against parameter-matched Transformer sequential fine-tuning baselines, not against the full continual-learning literature.

The rows labeled `Transformer Curriculum LR=...` represent Transformer curriculum runs with different learning rates. They do not use replay, EWC, adapters, or other explicit continual-learning methods.

## Curriculum Stages
The curriculum is designed to test whether a model can learn later code-generation distributions without overwriting earlier behavior.

- **Stage 2A:** Foundational Python syntax, simple rules, and small code-generation patterns.
- **Stage 2E:** Early-stage adversarial/holdout evaluation set used to measure whether basic syntax-level behavior survives later training.
- **Stage 3:** Paraphrased and semantic instruction-following tasks. This is the main intermediate retention target.
- **Stage 5:** Final teacher-generated full-function coding tasks used for specialization.

The headline experiment asks: after training through the final stage, how much Stage 3 behavior and Stage 2E behavior remain?

## Main Result: Curriculum Retention

### Results Table


| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 3.8% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **SamatNext v0.2-B** | Curriculum lr=3e-6 | 100.0% | **98.8%** | **12.0%** |

Full per-example artifacts are stored locally under `results/runs/` and are gitignored because they may be large. To reproduce them, run `make reproduce-main-table-fresh`. External artifact archive: GitHub Release v0.1.0-reproducibility.


### Interpretation
The strongest Transformer curriculum baseline retains very little Stage 3 behavior. SamatNext-v0.1 retains much more Stage 3 behavior.

This suggests a retention/plasticity tradeoff: the Transformer specializes more strongly on the final distribution, while SamatNext preserves more of the immediately preceding semantic stage. Stage 2E remains low for both models, so the result should not be framed as long-horizon forgetting being solved.

## Reproducibility
To run the standard verification tasks and pipelines, use the provided `Makefile` commands:
```bash
make setup
make test
make prepare-data-smoke
make bench-vram
make reproduce-smoke
make reproduce-main-table
make paper-check
```

To run a fresh evaluation run on model checkpoints and output full detailed artifacts, run:
```bash
make reproduce-main-table-fresh
```
Or directly:
```bash
python scripts/reproduce_main_table.py --force-eval --timeout-seconds 5 --output results/runs/fresh_eval_<timestamp>/
```

## Quick Start (Beginner Friendly)

You can run and explore **SamatNext v0.2-B** using either pre-trained weights or by training the model from scratch.

### Option 1: Load and Evaluate Pre-trained Weights (Recommended)
This option allows you to load our verified checkpoints and evaluate them immediately.

1. **Place Checkpoints:** Make sure the following checkpoints are placed in the `checkpoints/` directory:
   * `samatnext_v02b_stage2a.pt` (Stage 2A Best)
   * `samatnext_v02b_stage3_best.pt` (Stage 3 Best)
   * `samatnext_v02b_stage5_best.pt` (Stage 5 Best)
2. **Run Evaluation:** Run the hybrid evaluation suite to test the model across the curriculum tasks:
   ```bash
   python scripts/reproduce_main_table.py --output results/runs/official_v02b_eval/
   ```
3. **Programmatic Usage:** Load and run a forward pass in Python:
   ```python
   import torch
   from models.samat_next.config import SamatNextConfig
   from models.samat_next.model import SamatNextForCausalLM
   from transformers import AutoTokenizer

   # Load Config & Model
   config = SamatNextConfig.from_json("configs/ablations/samat_next_v0_2b_official.json")
   model = SamatNextForCausalLM(config)

   # Load Stage 5 Weights
   weights = torch.load("checkpoints/samatnext_v02b_stage5_best.pt", map_location="cpu")
   state_dict = weights["model_state_dict"] if "model_state_dict" in weights else weights
   model.load_state_dict(state_dict, strict=True)
   model.eval()

   # Tokenize Prompt and Generate
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
   prompt = "<|im_start|>user\nDefine a function add_two(x, y) that returns their sum.<|im_end|>\n<|im_start|>assistant\n"
   inputs = tok(prompt, return_tensors="pt").input_ids
   with torch.no_grad():
       logits, _ = model(inputs)
       print("Next token logit shape:", logits.shape)
   ```

### Option 2: Train the Model from Scratch (Do-It-Yourself)
Follow these steps to run the complete curriculum pre-training and sequential fine-tuning yourself:

1. **Prepare the Data:** Download and process the curriculum datasets:
   ```bash
   make prepare-data
   ```
2. **Train Stage 2A (Pretraining from Scratch):**
   ```bash
   python scripts/train_stage2a.py --config configs/ablations/samat_next_v0_2b_official.json --output-checkpoint-prefix samatnext_v02b_stage2a --batch-size 1 --grad-accum-steps 32
   ```
3. **Train Stage 3 (Sequential Fine-tuning for Paraphrase):**
   ```bash
   python scripts/train_stage3.py --config configs/ablations/samat_next_v0_2b_official.json --input-checkpoint checkpoints/samatnext_v02b_stage2a.pt --output-checkpoint-prefix samatnext_v02b_stage3
   ```
4. **Train Stage 5 (Teacher-Student Distillation):**
   ```bash
   python scripts/train_stage5.py --config configs/ablations/samat_next_v0_2b_official.json --input-checkpoint checkpoints/samatnext_v02b_stage3_best.pt --output-checkpoint-prefix samatnext_v02b_stage5
   ```


Paper source tag:
`v0.1.0-paper`

Reproducibility artifact commit:
`525665fe790b18668251dad6698fe9bfe0ca27ca`

External artifact archive:
GitHub Release [`v0.1.0-reproducibility`](https://github.com/samat2003/samatnext-v0.1/releases/tag/v0.1.0-reproducibility)

Paper status:
Technical report in preparation.


## Limitations
- The curriculum is synthetic and narrow, focused on Python code tasks.
- The headline result is not a broad HumanEval/MBPP/SWE-bench claim.
- Stage 2E retention remains low for both SamatNext and Transformer baselines.
- The current release does not yet include full architectural ablations separating Differential-Attention-style layers from DeltaNet-inspired simplified linear-state mixers.
- The Transformer baselines are parameter-matched and learning-rate-varied, but they are not compared against explicit continual-learning methods such as replay, EWC, or adapter isolation.
- The Transformer curriculum baseline is evaluated across multiple learning rates, while the current SamatNext curriculum result is reported at LR=3e-6. Additional SamatNext LR=1e-5 and LR=3e-5 runs are needed to fully isolate architecture effects from learning-rate effects.
- Some Stage 5 data is teacher-generated, so dataset provenance and licensing are documented separately.

## Licensing
- **First-party Source Code:** Licensed under the Apache-2.0 License.
- **Checkpoints:** Weights are licensed under the research-only Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Datasets:** See `DATA_LICENSES.md`.
- **Teacher-generated data:** Stage 5 examples derived from Qwen2.5-Coder-3B-Instruct outputs are treated as license-sensitive and documented separately.

## Safety
Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code.
