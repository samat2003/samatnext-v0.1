# Paper Draft Outline

## Title
SamatNext-v0.1: Hybrid Differential/Linear-State Decoders Improve Curriculum Retention in Small Code Models

## Abstract
Standard Transformer decoders tend to suffer from catastrophic forgetting when trained sequentially on shifting curriculum objectives. We present SamatNext-v0.1, an experimental hybrid sequence-mixing decoder that alternates Differential Attention layers with linear-state sequence mixers. We compare this architecture under a controlled sequential Python code curriculum against a parameter-matched Transformer baseline. Our findings suggest that the hybrid design displays stronger curriculum retention and sequential plasticity compared to the baseline under identical token budgets and training conditions.

## 1. Introduction
- Background: Seq2Seq and code models.
- Core Problem: Catastrophic forgetting under progressive curriculum learning.
- Thesis: Alternating linear mixers and differential attention provides a favorable inductive bias for sequential retention.

## 2. Related Work
- Linear sequence mixers (DeltaNet, Mamba).
- Multi-Head Attention variations and Differential Attention.
- Continual learning / curriculum retention in language models.

## 3. Method
- SamatNext architecture configuration.
- Alternating sequence mixer patterns.
- Differential attention mechanism.
- DeltaNet-inspired linear-state mixer details.

## 4. Experimental Setup
- Staged curriculum: Stage 2A (Syntax) -> Stage 3 (Semantics) -> Stage 5 (Teacher-Generated SFT).
- Baseline details: Parameter-matched Transformer (vocab size 151,936, intermediate size 2304).
- Identical tokenizer, optimizer, batch accumulation, and FP32 training settings.

## 5. Results
- Curriculum retention tables.
- Comparisons on held-out Stage 2E and Stage 3 datasets.

## 6. Ablations
- Positional encodings (No-RoPE vs RoPE in attention).
- Mixer pattern patterns (alternating, all-attention, all-linear).
- Verifier head presence.

## 7. Limitations
- Scale limits (<400M parameters).
- Language restrictions (Python only).
- Non-positional nature of the linear-state mixer path.

## 8. Reproducibility Appendix
- Code manifest hashes, hardware details, seed details.

---

## Paper License Decision

* **Preferred:** CC BY 4.0 for maximum reuse/open-science signaling.
* **Conservative option:** arXiv perpetual non-exclusive license.
* **Note:** This paper license is separate from the repository Apache-2.0 source-code license.
