# Known Bugs and Fixes

During the development and testing of SamatNext and the curriculum pipeline, several severe bugs were discovered and patched. This document catalogs them to ensure future maintainers understand why certain checks and balances exist.

### 1. Wrong Qwen Special Tokens
- **Symptom:** The model struggled to learn the prompt format and output garbage or refused to stop.
- **Root Cause:** The tokenizer config was initially missing the exact `Qwen2.5-Coder` special tokens `<|im_start|>` and `<|im_end|>`.
- **Fix:** Explicitly mapped the tokens and ensured the target labels included the `<|im_end|>` termination.
- **Affects Published Results?** No. This was fixed before the official Stage 5 comparisons.

### 2. All-Masked Attention NaN
- **Symptom:** Model loss exploded to NaN during prompt-only or heavily padded sequences.
- **Root Cause:** If an entire row of the causal attention matrix was masked out (e.g., in pad regions), the softmax denominator became 0.
- **Fix:** Handled extreme negative masking limits safely.
- **Affects Published Results?** No. This was fixed early in Stage 2.

### 3. Zero-Valid Target Labels
- **Symptom:** NaN gradients during batch updates.
- **Root Cause:** If the model was trained using instruction-masking (masking the prompt), and a batch happened to contain examples with extremely short targets or padding, it could result in zero valid unmasked labels, causing division-by-zero in the CrossEntropy loss.
- **Fix:** Masking logic updated to ensure at least one valid token is always present or the batch is skipped.
- **Affects Published Results?** No.

### 4. FP32 Stability Requirement
- **Symptom:** Occasional loss spikes or divergence late in training.
- **Root Cause:** The hybrid DifferentialAttention / DeltaNet recurrent logic relies on accumulations that easily overflow or underflow in pure FP16/BF16.
- **Fix:** Switched training to FP32, combined with `grad_clip=0.5`.
- **Affects Published Results?** No. All comparisons (Transformer and SamatNext) were run under the stable FP32 setup.

### 5. `<|im_end|>` vs `<|endoftext|>` Stop-Token Bug
- **Symptom:** False reports of high repetition rates during evaluation. The model generated the correct answer, but then appended garbage tokens.
- **Root Cause:** The eval generator was instructed to stop on `<|endoftext|>` (id 151643), but the ChatML format targets trained the model to stop with `<|im_end|>` (id 151645). The generator ignored `<|im_end|>` and forced generation up to `max_new_tokens`.
- **Fix:** Eval suite updated to stop on either token, strip trailing data, and compute metrics cleanly.
- **Affects Published Results?** No. The core Stage 2/3/5 eval runs did not suffer from this issue heavily due to the format logic used in those tests, and the bug was definitively patched prior to final validation.

### 6. Unsafe Raw `exec()` in Evaluation
- **Symptom:** Evaluation script would completely hang indefinitely.
- **Root Cause:** `exec()` was run directly in the main python thread. If the model generated `while True: pass`, the evaluation thread locked forever.
- **Fix:** All evaluation now executes inside a subprocess sandbox with a strict `1.0s` timeout limit.
- **Affects Published Results?** No. It merely delayed the evaluation workflow.

### 7. Stage 6 Completion Contamination
- **Symptom:** Model syntax collapse and extreme repetition rates during Stage 6 HumanEval-style bridging.
- **Root Cause:** Extracted completion datasets from HuggingFace contained test boilerplate, `if __name__` blocks, and corrupted ASTs. Furthermore, mixing full-function targets with body-only targets in the same batch caused severe objective confusion.
- **Fix:** Stage 6 was entirely abandoned and archived.
- **Affects Published Results?** No. All Stage 6 experiments were strictly quarantined and removed from the active curriculum.
