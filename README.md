# SamatNext-v0.1

**A hybrid sequence-mixing decoder alternating Differential Attention and linear-state mixers for curriculum retention and sequential plasticity.**

---

## Summary
SamatNext-v0.1 is an experimental language model architecture designed to investigate the sequence-modeling trade-offs between Multi-Head Attention and linear sequence mixers under staged curriculum training. The project explores the problem of catastrophic forgetting, where standard Transformers overwrite their understanding of early training distributions when trained sequentially on a curriculum of code objectives.

## Research Claim
> **Target Scientific Claim:**
> “SamatNext-v0.1, a hybrid decoder alternating differential-attention-like layers with linear-state sequence mixers, improves curriculum retention / sequential plasticity compared with a parameter-matched Transformer baseline under controlled small-scale Python code-model experiments.”

Our findings suggest that alternating recurrent/state-tracking sequence mixers with Differential Attention provides a favorable inductive bias that preserves prior knowledge without requiring complex rehearsal, parameter gating, or custom regularization.

## What This Is / Is Not
### What This Is:
- An experimental research prototype (~350M parameters).
- A controlled scientific ablation studying curriculum learning in code models.
- A fully reproducible workflow with exact parameter matching and execution sandboxes.

### What This Is NOT:
- **Not** a general SOTA frontier coding assistant.
- **Not** production-ready.
- **Not** a general replacement for standard Transformers across all NLP tasks.
- **Not** trained or evaluated on complete HumanEval for benchmarks (exploratory HumanEval-5 subset results are archived and should not be treated as a full benchmark; HumanEval is completely held out of all SFT training data).

## Architecture
The **SamatNext-v0.1** autoregressive next-token decoder integrates:
- **Differential Attention Layers**: Alternating hybrid attention that improves signal-to-noise ratio by subtracting two distinct attention keys/queries.
- **DeltaNet-style Recurrent Layers**: Linear-state tracking mixers that maintain state representation across context boundaries.
- **Feed-Forward Blocks**: Standard SwiGLU non-linear mappings.
- **Normalization**: RMSNorm layers with a configurable epsilon of $1\times 10^{-6}$.
- **Formatting**: Qwen-style chat syntax structure (`<|im_start|>`, `<|im_end|>`).

*Note on Vocabulary vs. Tokenizer Length: The configuration sets the model vocabulary size to 151,936 to match the standard Qwen2.5-Coder architecture embedding dimensions, whereas the tokenizer itself actually defines 151,665 active tokens. The remaining 271 indices are reserved or unused.*

## Matched Transformer Baseline
To ensure a fair baseline comparison, the Transformer baseline is parameter-matched to SamatNext within a strict 0.01% parameter count tolerance. Both models share the identical vocabulary structure, intermediate MLP dims, and layer counts, trained under identical optimizers, learning rates, sequence lengths, and batch scheduling.

## Main Result
Under our controlled staged curriculum (Stage 2A syntax $\rightarrow$ Stage 3 semantics $\rightarrow$ Stage 5 SFT Sprints), SamatNext demonstrates significantly higher retention of earlier stages compared to the Transformer baseline.

### Results Table

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 83.0% | **70.2%** | **4.3%** |
*Note: This table was generated from a fresh evaluation run on 2026-06-12 09:46:46. Full per-example artifacts are saved in results/runs/fresh_eval_20260612_094500.*

## Parameter Matching
The parameter matching checker strictly validates the configuration profiles:
- **SamatNext total parameters:** 356,082,440
- **Transformer baseline parameters:** 356,082,432
- **Exact Difference:** 8 parameters (0.000002%)

Detailed structural parameter breakdown is maintained in `results/tables/parameter_counts.md`.

## VRAM Profile
Peak GPU memory usage during evaluation (batch size 1, sequence length up to 2048) on an NVIDIA GeForce RTX 5070 Ti Laptop GPU:

| Model Type | 128 context | 512 context | 1024 context | 2048 context |
| :--- | :---: | :---: | :---: | :---: |
| **SamatNext-v0.1** | 2819.52 MB | 4418.96 MB | 5624.84 MB | 8007.85 MB |
| **Matched Transformer** | 2815.63 MB | 4537.20 MB | 6213.71 MB | 10770.97 MB |

## Evaluation Protocol
- **Greedy Decoding:** Temperature 0.0, Argmax selection.
- **Subprocess Isolation:** Generated code is executed inside a sandboxed subprocess with strict CPU/memory limits and timeout gates.
- **Metrics:** Strict AST parsing checks syntax status, and execution assertions measure pass/fail rates.

## Reproducibility Commands
To run the standard verification tasks and pipelines, use the provided `Makefile` commands:

1. **Environment Setup:**
   ```bash
   make setup
   ```
2. **Run Test Suite:**
   ```bash
   make test
   ```
3. **Build Smoke Datasets:**
   ```bash
   make prepare-data-smoke
   ```
4. **Profile VRAM Limits:**
   ```bash
   make bench-vram
   ```
5. **Run E2E Smoke Pipeline:**
   ```bash
   make reproduce-smoke
   ```
6. **Regenerate Tables from Cache:**
   ```bash
   make reproduce-main-table
   ```
7. **Verify Checklist Artifacts:**
   ```bash
   make paper-check
   ```

*Note on Artifact Storage: The `results/runs/` folder is gitignored due to the large size of detailed per-example JSON files. Full fresh evaluation artifacts can be regenerated locally using `python scripts/reproduce_main_table.py --force-eval`, or downloaded from the external release archive.*

## Data / Contamination
Training data is strictly filtered for contamination against HumanEval and MBPP via AST-level subtree matching and exact string overlays.
- **Contamination Report:** See `reports/contamination_report.md` for detail.
- **Licenses:** See `DATA_LICENSES.md` for details on Flytech, CodeFeedback, and Qwen teacher-distilled datasets.

## Licensing
The repository follows a strict multi-license boundary configuration:
- **First-party Source Code:** Licensed under the [Apache-2.0 License](LICENSE).
- **Checkpoints:** Weights are licensed under the research-only [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](CHECKPOINT_LICENSE.md).
- **Data Assets:** Subject to dataset-specific terms (see [DATA_LICENSES.md](DATA_LICENSES.md)).
- **Upstream Code & Tokenizers:** Licensed under original developer terms (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **Paper Draft / Outline:** Subject to CC-BY-4.0.

## Safety
Running code evaluations executes untrusted model-generated scripts. The execution runner implements subprocess sandboxing, but it is not a complete virtualization boundary. Do not run evaluations on systems with access to sensitive credentials or networks. For safety details, see [SECURITY.md](SECURITY.md).

## Limitations
- **Scaling Uncertainty:** Study is limited to small-scale configurations (<400M params).
- **Language Scope:** Restricted exclusively to Python.
- **Recurrence Constraint:** DeltaNet recurrent sequence-mixing steps do not incorporate positional encodings, relying entirely on hidden state sequence evolution.

## Citation
If you use this work or refer to the SamatNext curriculum retention results, please cite:
```bibtex
@software{samatnext2026reproducibility,
  author       = {Samat Zharassov},
  title        = {SamatNext-v0.1: A Reproducible Hybrid sequence-mixing coding model curriculum experiment},
  year         = 2026,
  url          = {https://github.com/samat2003/samatnext-v0.1}
}
```
