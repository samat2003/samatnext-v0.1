# Paper Draft Outline

## Title
SamatNext v0.2-B: An Exploratory Study of RMS-Normalized Hybrid Decoders for Curriculum Retention in Small Code Models

## Abstract
Standard Transformer decoders can exhibit substantial forgetting under sequential fine-tuning when trained sequentially on shifting curriculum objectives. We present SamatNext v0.2-B, an experimental hybrid sequence-mixing decoder that alternates Differential Attention layers with linear-state sequence mixers featuring RMS Normalization and Output Scale Calibration. We compare this architecture under a controlled sequential Python code curriculum against a parameter-matched Transformer baseline. Our findings suggest that under this controlled setting, the hybrid design with scale calibration achieves a 100.0% pass rate on the controlled Stage 5 holdout while retaining 98.8% of adjacent semantic stage capabilities under identical token budgets. However, it does not completely solve catastrophic forgetting on earlier curriculum stages.

## 1. Introduction
- Background: Seq2Seq and code models.
- Core Problem: Catastrophic forgetting under progressive curriculum learning.
- Thesis: Alternating linear mixers with RMS-normalized scale calibration and differential attention provides a robust inductive bias for sequential retention.

## 2. Related Work
- Linear sequence mixers (DeltaNet, Mamba).
- Multi-Head Attention variations and Differential Attention.
- Continual learning / curriculum retention in language models.

## 3. Method
- SamatNext v0.2-B architecture configuration.
- Alternating sequence mixer patterns.
- Differential attention mechanism.
- DeltaNet-inspired linear-state mixer details.
- RMS Normalization and Output Scale Calibration (0.25 scaling) on LSM outputs.
- Verifier head projection (`use_verifier_head = true`).

## 4. Experimental Setup
- Staged curriculum: Stage 2A (Syntax) -> Stage 3 (Semantics) -> Stage 5 (Teacher-Generated SFT).
- Baseline details: Parameter-matched Transformer (vocab size 151,936, intermediate size 2304).
- Identical tokenizer, optimizer, batch accumulation, and FP32 training settings.

## 5. Results
- Curriculum retention tables.
- Comparisons on held-out Stage 2E and Stage 3 datasets.
- Verification of 100.0% Stage 5 Pass, 98.8% Stage 3 Retention, and 12.0% Stage 2E Pass.

## 6. Ablations
- Positional encodings (No-RoPE vs RoPE in attention).
- Mixer pattern patterns (alternating, all-attention, all-linear).
- Verifier head presence.
- LSM Output Scale tuning (0.05, 0.10, 0.25, 0.50).

## 7. Limitations
- Scale limits (~356M parameters).
- Language restrictions (Python only).
- Non-positional nature of the linear-state mixer path.

## 8. Reproducibility Appendix
- Code manifest hashes, hardware details, seed details.

---

## Paper License Decision

* **Preferred:** CC BY 4.0 for maximum reuse/open-science signaling.
* **Conservative option:** arXiv perpetual non-exclusive license.
* **Note:** This paper license is separate from the repository Apache-2.0 source-code license.
