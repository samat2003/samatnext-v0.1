# SamatNext-v0.1: Curriculum Retention and Sequential Plasticity

## Experiment Goal
This experiment investigates the "curriculum retention" properties of **SamatNext-v0.1**, a custom experimental coding-model architecture (~356M parameters), compared against a matched standard Transformer baseline (~346M parameters). We aim to see if SamatNext's hybrid architecture exhibits stronger sequential plasticity (learning new tasks without catastrophically forgetting earlier tasks) under a controlled curriculum setup.

## Architecture Summary
**SamatNext-v0.1** modifies the standard Transformer decoder by incorporating:
- **DifferentialAttention**: A hybrid attention mechanism.
- **DeltaNet-style Recurrent/State Layers**: To improve sequential state tracking.
- Qwen-style chat formatting (`<|im_start|>`, `<|im_end|>`).
- Standard Embedding, RMSNorm, and FFN (SwiGLU) blocks.

**Transformer Baseline**: A matched decoder-only architecture utilizing standard Multi-Head Attention without DeltaNet augmentations.

## Dataset Stages
The curriculum consisted of the following sequential stages:
1. **Stage 2A (Syntax & Rules)**: Basic Python syntax, rules, and small algorithms.
2. **Stage 3 (Paraphrase & Semantics)**: Paraphrased instructions to encourage semantic robustness over rote memorization.
3. **Stage 5 (Teacher Distillation)**: High-quality instructional full-function distillation from `Qwen2.5-Coder-3B-Instruct`.
- *(Holdout)* **Stage 2E (Adversarial)**: Edge cases and trick questions to test robustness.

## Training Setup
- **Curriculum Training**: The models were trained sequentially from scratch -> Stage 2A -> Stage 3 -> Stage 5. 
- **Learning Rate Rescue**: To test if the Transformer's catastrophic forgetting was merely a learning rate failure, we ran "rescue" ablations by drastically lowering the learning rate (from `3e-4` down to `1e-5` and `3e-6`) during Stage 5.

## Final Comparison Table

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
|---|---|---|---|---|
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 1.0% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 97.6% | **86.8%** | **6.3%** |

## Interpretation
Under this controlled setup, SamatNext-v0.1 demonstrated massive improvements in Stage 3 Retention (86.8%) compared to the best Transformer baseline rescue attempt (6.0%). The Transformer suffered catastrophic forgetting of earlier stages the moment it successfully converged on Stage 5. 

This is **evidence of stronger curriculum-retention / sequential-plasticity behavior** under this specific setup. 
**It is NOT proof of general coding superiority.** It is NOT proof that the architecture beats Transformers in all settings or at massive scale.

## Limitations
- **Small Scale**: The models are <400M parameters. These results may not extrapolate to 7B+ scale.
- **Narrow Curriculum**: The domain is restricted to short, highly-controlled Python coding tasks.
- **Experimental Eval**: Evaluation was done using strict string matching and subprocess isolation with timeout and resource limits. This is not a secure sandbox.

## Why Stage 6B/6C are Excluded
Attempts to extend this curriculum into "Stage 6" (completion-only, HumanEval-style targets) were abandoned. The datasets used for completion training contained corrupted targets, doctest contamination, and caused severe metric instability. Because the eval `<|im_end|>` stop-token bug conflated format mismatch with catastrophic forgetting, all Stage 6 results have been archived and are excluded from architecture evidence.
