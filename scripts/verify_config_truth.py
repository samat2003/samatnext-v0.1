import os
import sys
import torch
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
CKPT_PATH = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")
CONFIG_PATH = os.path.join(ROOT, "configs", "samat_next_v0_1.json")

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return
        
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        return
        
    config = SamatNextConfig.from_json(CONFIG_PATH)
    model = SamatNextForCausalLM(config)
    
    # Load state dict strictly to ensure match
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n--- MODEL CONFIG TRUTH ---")
    print(f"Checkpoint Path: {CKPT_PATH}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Num Layers: {getattr(config, 'num_layers', getattr(config, 'num_hidden_layers', 'unknown'))}")
    print(f"Attention Heads: {config.num_attention_heads}")
    print(f"KV Heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    print(f"FFN Size: {config.intermediate_size}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Exact Parameter Count: {total_params:,}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
