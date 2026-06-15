# SamatNext v0.2-B Standalone Model Release

This branch contains the standalone model definition and configuration for **SamatNext v0.2-B**, a ~350M-parameter autoregressive language model incorporating:
- **Differential-Attention-style layers**
- **DeltaNet-inspired Linear State Mixers (LSM)**
- **RMS Normalization + Scale Calibration** on the LSM outputs (`use_lsm_rmsnorm = true`, `lsm_output_scale = 0.25`).

Unlike the main branch, this branch is stripped of all training/evaluation scaffolding, datasets, and logs, leaving only the clean model architecture for easy integration.

## Performance vs. Baseline Transformer (Curriculum Benchmark)

SamatNext v0.2-B was evaluated head-to-head against a parameter-matched Transformer baseline across a multi-stage Python curriculum to test **catastrophic forgetting**. The benchmark measures how much previous knowledge is retained after learning new tasks:

| Model | Training Path | Stage 5 Pass (New Task) | Stage 3 Retention (Old Task) | Stage 2E Pass (Syntax) |
| :--- | :--- | :---: | :---: | :---: |
| **SamatNext v0.2-B** | Curriculum lr=3e-6 | **100.0%** | **98.8%** | **12.3%** |
| **Transformer** | Curriculum Rescue lr=1e-5 | 97.6% | 6.0% | 3.0% |
| **Transformer** | Curriculum lr=3e-6 | 49.4% | 4.0% | 0.0% |

### Key Takeaway
Standard Transformers suffer from severe catastrophic forgetting during curriculum training—retaining less than **6.0%** of their prior reasoning skills. By contrast, **SamatNext v0.2-B** achieves a perfect **100.0% Pass Rate** on the new task while retaining **98.8%** of its previous intermediate capabilities, demonstrating a near-perfect sequential plasticity trade-off.

## Installation

Ensure you have Python 3.8+ and install the dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start (Loading the Model)

### Option 1: Load with Pre-trained Weights (Recommended)
To load the model with the pre-trained weights from Stage 5 (Teacher-Student Distillation):

1. **Download the Weights:** Download `samatnext_v02b_stage5_best.pt` from the releases page and place it in your working directory.
2. **Load and Run:**
   ```python
   import torch
   from models.samat_next.config import SamatNextConfig
   from models.samat_next.model import SamatNextForCausalLM
   from transformers import AutoTokenizer

   # 1. Load the model configuration
   config = SamatNextConfig.from_json("samat_next_v0_2b_config.json")
   model = SamatNextForCausalLM(config)

   # 2. Load weights
   weights = torch.load("samatnext_v02b_stage5_best.pt", map_location="cpu")
   state_dict = weights["model_state_dict"] if "model_state_dict" in weights else weights
   model.load_state_dict(state_dict, strict=True)
   model.eval()

   # 3. Tokenize input prompt
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
   prompt = "<|im_start|>user\nWrite a Python function to compute factorial.<|im_end|>\n<|im_start|>assistant\n"
   inputs = tokenizer(prompt, return_tensors="pt").input_ids

   # 4. Generate code
   with torch.no_grad():
       # Forward pass to get logits
       logits, _ = model(inputs)
       print("Output logits shape:", logits.shape)
   ```

### Option 2: Load as a Scratch Model (Random Weights)
If you want to initialize the model from scratch (with random weights) to train it on your own datasets:
```python
from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

# Load config and instantiate
config = SamatNextConfig.from_json("samat_next_v0_2b_config.json")
model = SamatNextForCausalLM(config)

print(f"Instantiated scratch SamatNext model with {sum(p.numel() for p in model.parameters()):,} parameters.")
```

## Model Architecture Code
The architecture code is located under the `models/samat_next/` folder:
- `model.py`: Core autoregressive next-token decoder definition.
- `config.py`: Architecture configuration settings.
- `linear_state_mixer.py`: Linear State Mixer (LSM) layer logic.
- `differential_attention.py`: Differential attention mechanism.

## License
Model code is licensed under the Apache-2.0 License. Model weights are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

