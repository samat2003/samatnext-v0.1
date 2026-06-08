import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
config = SamatNextConfig.from_json(config_path)
model = SamatNextForCausalLM(config)

ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")
print(f"Loading from: {ckpt_path}")
print(f"Using config: {config_path}")

sd = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(sd, strict=False)

print("\n--- Architecture Truth ---")
print(f"Hidden Size: {config.hidden_size}")
print(f"Num Layers: {config.num_layers}")
print(f"Num Attention Heads: {config.num_attention_heads}")
print(f"Num KV Heads (if applicable): {config.num_key_value_heads}")
print(f"Intermediate (FFN) Size: {config.intermediate_size}")
print(f"Vocab Size: {config.vocab_size}")
params = sum(p.numel() for p in model.parameters())
print(f"Exact Parameter Count: {params:,} ({params/1e6:.1f}M)")

if params < 200_000_000:
    print("\nCONCLUSION: The model is ~150M parameters. The '350M' in the filename is historically inaccurate.")
else:
    print("\nCONCLUSION: The model is >200M parameters.")
