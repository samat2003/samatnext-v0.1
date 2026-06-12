# SamatNext-v0.1

[![CI](https://github.com/samat2003/samatnext-v0.1/actions/workflows/ci.yml/badge.svg)](https://github.com/samat2003/samatnext-v0.1/actions/workflows/ci.yml)

## Summary
SamatNext-v0.1 is an experimental ~350M-parameter autoregressive language model architecture designed to study retention under staged curriculum training for Python code tasks. The project compares a hybrid decoder using Differential-Attention-style layers and DeltaNet-inspired linear-state mixers against a parameter-matched Transformer baseline.

## Research Claim
SamatNext-v0.1 shows substantially stronger intermediate semantic curriculum retention than a parameter-matched Transformer sequential fine-tuning baseline under a controlled Python curriculum, while sacrificing some final-stage specialization.

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
- **DeltaNet-inspired linear-state mixer layers:** Recurrent/linear-state sequence-mixing blocks intended to test whether state-tracking inductive bias affects curriculum retention.
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
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum LR=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum LR=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum LR=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum LR=3e-6 | 83.0% | 70.2% | 4.3% |

Full per-example artifacts are stored locally under `results/runs/` and are gitignored because they may be large. To reproduce them, run `make reproduce-main-table-fresh`.

### Interpretation
The strongest Transformer curriculum baseline reaches higher final Stage 5 performance, but retains very little Stage 3 behavior. SamatNext-v0.1 reaches lower final Stage 5 performance but retains much more Stage 3 behavior.

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
- The current release does not yet include full architectural ablations separating Differential-Attention-style layers from DeltaNet-inspired linear-state mixers.
- The Transformer baselines are parameter-matched and learning-rate-varied, but they are not compared against explicit continual-learning methods such as replay, EWC, or adapter isolation.
- Some Stage 5 data is teacher-generated, so dataset provenance and licensing are documented separately.

## Licensing
- **First-party Source Code:** Licensed under the Apache-2.0 License.
- **Checkpoints:** Weights are licensed under the research-only Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Datasets:** See `DATA_LICENSES.md`.
- **Teacher-generated data:** Stage 5 examples derived from Qwen2.5-Coder-3B-Instruct outputs are treated as license-sensitive and documented separately.

## Safety
Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code.
