# SPDX-License-Identifier: Apache-2.0
import os
import sys
import torch
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="samatnext", choices=["samatnext", "transformer"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    # Set defaults based on model type
    if args.model == "samatnext":
        config_path = args.config if args.config else os.path.join(ROOT, "configs", "samatnext_350m.json")
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")
    else:
        config_path = args.config if args.config else os.path.join(ROOT, "configs", "transformer_350m_matched.json")
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_stage5_best.pt")
        
    print(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
        
    if args.model == "samatnext":
        config = SamatNextConfig.from_json(config_path)
        model = SamatNextForCausalLM(config)
    else:
        config = TransformerConfig.from_json(config_path)
        model = TransformerForCausalLM(config)
        
    print(f"Model: {args.model.upper()}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Num Layers: {config.num_layers}")
    print(f"Attention Heads: {config.num_attention_heads}")
    print(f"FFN Size: {config.intermediate_size}")
    print(f"Vocab Size: {config.vocab_size}")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        important_missing = [k for k in missing_keys if "freqs_cis" not in k]
        important_unexpected = [k for k in unexpected_keys if "freqs_cis" not in k]
        if important_missing:
            raise RuntimeError(f"Missing key(s) in state_dict: {important_missing}")
        if important_unexpected:
            raise RuntimeError(f"Unexpected key(s) in state_dict: {important_unexpected}")
        print("Checkpoint loaded successfully (matched parameters).")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path} (skipping checkpoint loading)")
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Exact Parameter Count: {total_params:,}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
