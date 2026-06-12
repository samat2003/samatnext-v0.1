# SamatNext-v0.1

## Summary
SamatNext-v0.1 is an experimental language model architecture designed to investigate the sequence-modeling trade-offs between Multi-Head Attention and linear sequence mixers under staged curriculum training. The project explores the problem of catastrophic forgetting, where standard Transformers overwrite their understanding of early training distributions when trained sequentially on a curriculum of code objectives.

## Research Claim
SamatNext-v0.1 shows substantially stronger semantic curriculum retention than a parameter-matched Transformer baseline under a controlled sequential Python curriculum, while sacrificing some final-stage specialization.

## What This Is / Is Not
### What This Is:
- An experimental research prototype (~350M parameters).
- A controlled scientific ablation studying curriculum learning in code models.
- A fully reproducible workflow with exact parameter matching and subprocess isolation.

### What This Is NOT:
- **Not** a general SOTA frontier coding assistant.
- **Not** production-ready.
- **Not** a general replacement for standard Transformers across all NLP tasks.
- **Not** trained or evaluated on complete HumanEval for benchmarks. HumanEval is never used for training. The repository includes held-out HumanEval evaluation infrastructure, but the headline result is the controlled Stage 2A → Stage 3 → Stage 5 curriculum-retention experiment.
- **Not** proven for billion-scale superiority.

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

## Main Result: Curriculum Retention

### Results Table


| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 83.0% | **70.2%** | **4.3%** |

Full per-example artifacts are stored locally under `results/runs/` and are gitignored because they may be large. To reproduce them, run `make reproduce-main-table-fresh`. External artifact archive: pending.


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
- **Subprocess Isolation:** Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code.
- **Metrics:** AST parsing checks syntax status, and execution assertions measure pass/fail rates.

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

Full fresh evaluation artifacts can be regenerated locally using `python scripts/reproduce_main_table.py --force-eval --timeout-seconds 5 --output results/runs/fresh_eval_<timestamp>/` (or via `make reproduce-main-table-fresh`). The `results/runs/` directory is gitignored because per-example artifacts may be large. External artifact archive: pending.

## Data and Contamination Control
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
Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code.

Do not run evaluations on systems with access to sensitive credentials or networks. For safety details, see [SECURITY.md](SECURITY.md).

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

## Repository Layout
The key files and directories are organized as follows:
- `checkpoints/`: Model checkpoint weights.
- `configs/`: Model and architecture configuration JSONs.
- `data/`: Datasets and preprocessing manifests.
- `models/`: SamatNext and baseline Transformer architecture implementation.
- `paper/`: Outline and manuscripts for paper submission.
- `results/`: Cached results, tables, and run logs.
- `scripts/`: Training, data preparation, validation, and evaluation scripts.
- `tests/`: Pytest unit and regression tests.
- `Makefile`: Shortcut commands for standard pipeline tasks.
- `DATA_LICENSES.md`: Licensing and terms of training/eval datasets.
- `CHECKPOINT_LICENSE.md`: Non-commercial license terms for model weights.
- `MODEL_CARD.md`: Model architecture, training details, and limitations card.
- `SECURITY.md`: Safety guidance on running code evaluations.
