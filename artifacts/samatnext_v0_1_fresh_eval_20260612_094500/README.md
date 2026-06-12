# SamatNext-v0.1

## Summary
SamatNext-v0.1 is an experimental language model architecture designed to investigate the sequence-modeling trade-offs between Multi-Head Attention and linear sequence mixers under staged curriculum training.

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
- **Not** trained or evaluated on complete HumanEval for benchmarks.

## Architecture
The **SamatNext-v0.1** autoregressive next-token decoder integrates:
- **Differential Attention Layers**: Alternating hybrid attention that improves signal-to-noise ratio.
- **DeltaNet-style Recurrent Layers**: Linear-state tracking mixers that maintain state representation.
- **Feed-Forward Blocks**: Standard SwiGLU non-linear mappings.

## Matched Transformer Baseline
To ensure a fair baseline comparison, the Transformer baseline is parameter-matched to SamatNext within a strict 0.01% parameter count tolerance.

## Main Result: Curriculum Retention

### Results Table

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 83.0% | 70.2% | 4.3% |

Full per-example artifacts are stored locally under `results/runs/` and are gitignored because they may be large. To reproduce them, run `make reproduce-main-table-fresh`.

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

## Licensing
- **First-party Source Code:** Licensed under the Apache-2.0 License.
- **Checkpoints:** Weights are licensed under the research-only Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

## Safety
Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code.

External artifact archive: pending.
