# SamatNext-v0.1: Curriculum Retention and Sequential Plasticity

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 3.8% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **SamatNext v0.2-B** | Curriculum lr=3e-6 | 100.0% | **98.8%** | **12.0%** |

Full per-example artifacts are stored locally under `results/runs/` and are gitignored because they may be large. To reproduce them, run `make reproduce-main-table-fresh`. External artifact archive: GitHub Release v0.1.0-reproducibility.

## Out-of-Distribution Coding Smoke Test (HumanEval Subset)

| Model | Training Path | HumanEval Subset Pass@1 |
| :--- | :--- | :---: |
| **Transformer** | Curriculum lr=3e-6 | 8.0% |
| **SamatNext v0.2-B** | Curriculum lr=3e-6 | **12.0%** |
