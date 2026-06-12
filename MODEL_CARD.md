# Model Card: SamatNext-v0.1

## Model Details
- **Architecture:** Hybrid Decoder alternating Differential Attention layers and linear-state mixers (DeltaNet-inspired).
- **Parameters:** ~356 Million (exact parameter matched to Transformer baseline within 8 parameters).
- **Context Length:** 8,192 (trained on 512 context size for curriculum steps).
- **Vocabulary Size:** 151,936.
- **Language:** Python source code.
- **License:** Model code is licensed under Apache-2.0. Checkpoint weights are subject to the terms in `CHECKPOINT_LICENSE.md`.

## Intended Use & Research Scope
SamatNext-v0.1 is an experimental research model intended to study whether hybrid sequence-mixing architectures improve retention under staged Python code curricula. It is not intended as a production coding assistant, not claimed to be SOTA, and not claimed to outperform Transformers universally.

## Known Limitations
- **Scale Limits:** Under 400M parameter model scale; findings may not extrapolate to 7B+ scales.
- **Task Domain:** Optimized exclusively for Python code curricula. Short sequences (<512 tokens) only.
- **Benchmarking Limits:** Evaluated on controlled, decontaminated splits.

## Contamination Filtering & Safety
- **HumanEval / MBPP:** Explicitly held out from all training paths. Contamination tests are run prior to training.
- **Safety Warning:** Evaluation executes generated code under subprocess isolation. Subprocess isolation is not a complete security boundary. Never run evaluations of untrusted models on sensitive systems.
