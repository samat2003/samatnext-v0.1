# Model Card: SamatNext-v0.1

## Model Details
- **Architecture:** Hybrid Decoder alternating Differential Attention layers and linear-state mixers (DeltaNet-inspired).
- **Parameters:** ~356 Million (exact parameter matched to Transformer baseline within 8 parameters).
- **Context Length:** 8,192 (trained on 512 context size for curriculum steps).
- **Vocabulary Size:** 151,936 (configured to match standard Qwen2.5-Coder embedding dimensions; the tokenizer itself has 151,665 active tokens, with 271 reserved/unused).
- **Language:** Python source code.
- **License:** Model source code is licensed under Apache-2.0. Model checkpoint weights are separately licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see [CHECKPOINT_LICENSE.md](CHECKPOINT_LICENSE.md)).

## Intended Use & Research Scope
SamatNext-v0.1 is an experimental research model intended to study whether hybrid sequence-mixing architectures improve retention under staged Python code curricula. It is not intended as a production coding assistant, not claimed to be SOTA, and not claimed to outperform Transformers universally.

## Known Limitations
- **Scale Limits:** Under 400M parameter model scale; findings may not extrapolate to 7B+ scales.
- **Task Domain:** Optimized exclusively for Python code curricula. Short sequences (<512 tokens) only.
- **Benchmarking Limits:** Evaluated on controlled, decontaminated splits.

## Contamination Filtering & Safety
- **HumanEval / MBPP:** Explicitly held out from all training paths. Contamination tests are run prior to training.
- **Safety Warning:** Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or code. This is not a secure sandbox.
