# SamatNext-v0.1: Curriculum Retention and Sequential Plasticity

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 97.6% | **86.8%** | **6.3%** |
