# SamatNext-v0.1

**A custom small coding-model architecture exploring curriculum retention and sequential plasticity using hybrid DifferentialAttention and DeltaNet-style layers.**

## What This Is
- An experimental research repository.
- A custom small coding-model architecture (~350M parameters).
- An exploration of curriculum learning, specifically testing "sequential plasticity" (the ability to learn new tasks without catastrophically forgetting previous tasks).
- A rigorous comparison against a matched Transformer baseline under controlled conditions.

## What This Is NOT
- This is **NOT** a frontier coding model.
- This is **NOT** production-ready.
- This is **NOT** proven generally superior to Transformers in all domains.
- **HumanEval is never used for training.** Full HumanEval evaluation is pending / optional. Earlier exploratory HumanEval-5 subset results are archived and should not be treated as a full benchmark (all completion-only HumanEval-style experiments were abandoned due to format contamination).

## Architecture Summary
The **SamatNext-v0.1** architecture is an autoregressive next-token decoder combining:
- Standard Tokenization (151,936 vocabulary config, with 151,665 active tokenizer tokens; see note below), Embeddings, and LM Head.
- Standard RMSNorm and SwiGLU Feed-Forward Networks.
- **DifferentialAttention Layers**: An experimental hybrid attention mechanism.
- **DeltaNet-style Layers**: Recurrent/state-tracking layers to enhance sequential dependency learning.
- Qwen-style chat formatting (`<|im_start|>`, `<|im_end|>`).

*Note on Vocabulary vs. Tokenizer Length: The configuration sets the model vocabulary size to 151,936 to match the standard Qwen2.5-Coder architecture embedding dimensions, whereas the tokenizer itself actually defines 151,665 active tokens. The remaining 271 indices are reserved or unused.*

## Main Result: Curriculum Retention Advantage
The core hypothesis was that standard Transformers suffer catastrophic forgetting when traversing a curriculum of small datasets, effectively overwriting their understanding of early syntax rules once they memorize a later, more complex dataset. 

Under our specific, controlled small-model curriculum setup (Stage 2A -> Stage 3 -> Stage 5), **SamatNext showed drastically stronger retention than a matched Transformer baseline.**

### Results Table

| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |
| :--- | :--- | :---: | :---: | :---: |
| **Transformer** | Scratch → Stage5 | 97.6% | 0.8% | 3.3% |
| **SamatNext** | Scratch → Stage5 | 97.6% | 0.8% | 1.3% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum Rescue lr=3e-5 | 97.6% | 3.2% | 2.0% |
| **SamatNext** | Curriculum lr=3e-6 | 97.6% | **86.8%** | **6.3%** |

### Correct Interpretation
This is evidence of stronger curriculum-retention / sequential-plasticity behavior **under this setup**. It highlights a potential structural flaw in how standard Transformers handle sequential objective shifts. It is **not** proof of general coding superiority, nor does it guarantee the architecture scales better than Transformers to billions of parameters.

## Reproducibility
To run the automated tests, check the paper checklist, or run the reproducibility pipelines:

- **Run Verification Tests:** `python -m pytest tests/`
- **Verify Paper Checklist:** `python scripts/paper_check.py`
- **Run Smoke Test Pipeline:** `python scripts/reproduce_smoke.py`
- **Recreate Main Table & Update README:** `python scripts/reproduce_main_table.py`


## Future Work
- **Clean Full-Function Coding Generalization:** Training on strictly decontaminated Hugging Face Python datasets.
- **Completion-Only Interface:** Carefully constructing a HumanEval-style prompt-to-body interface without corrupting the objective function.
- **Larger Scale Ablations:** Testing the DeltaNet/DifferentialAttention hybrid at 1B+ scale to verify if the plasticity advantage holds.

For deeper technical details, please see the `reports/` and `checkpoints/` documentation.
