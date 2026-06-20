# -*- coding: utf-8 -*-
"""
layer_rms_diagnostic.py
========================
Logs per-layer hidden-state RMS for a single dummy batch through SamatNext.
Measures out.pow(2).mean(dim=-1).sqrt() at each block output.
Does NOT modify model behavior — read-only diagnostic.

Usage:
    python scripts/layer_rms_diagnostic.py --config configs/samatnext_350m.json --output results/runs/smoke_v02_lsm_rmsnorm/layer_rms_diagnostic.json
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.samat_next.differential_attention import DifferentialAttention
from models.samat_next.linear_state_mixer import DeltaNetInspiredLinearStateMixer


def diagnose(config_path, output_path, seed=42, seq_len=32, batch_size=1):
    torch.manual_seed(seed)
    cfg = SamatNextConfig.from_json(config_path)
    model = SamatNextForCausalLM(cfg)
    model.eval()

    dummy = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    layer_data = []

    with torch.no_grad():
        # Step through the model manually to capture per-layer hidden RMS
        h = model.model.embed_tokens(dummy)
        freqs_cis = model.model.freqs_cis[:seq_len] if model.model.freqs_cis is not None else None

        embed_rms = h.pow(2).mean(dim=-1).sqrt().mean().item()
        layer_data.append({
            "layer": "embedding",
            "type": "embedding",
            "hidden_rms_mean": round(embed_rms, 6),
        })

        for i, layer in enumerate(model.model.layers):
            mixer_type = "DifferentialAttention" if isinstance(layer.mixer, DifferentialAttention) else "LinearStateMixer"
            h = layer(h, freqs_cis)
            rms = h.pow(2).mean(dim=-1).sqrt()  # (batch, seq_len)
            layer_data.append({
                "layer": i,
                "type": mixer_type,
                "hidden_rms_mean": round(rms.mean().item(), 6),
                "hidden_rms_min": round(rms.min().item(), 6),
                "hidden_rms_max": round(rms.max().item(), 6),
                "hidden_rms_std": round(rms.std().item(), 6),
            })

        h = model.model.norm(h)
        final_rms = h.pow(2).mean(dim=-1).sqrt().mean().item()
        layer_data.append({
            "layer": "final_norm",
            "type": "RMSNorm",
            "hidden_rms_mean": round(final_rms, 6),
        })

    result = {
        "config_path": config_path,
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": cfg.num_layers,
        "hidden_size": cfg.hidden_size,
        "param_count": sum(p.numel() for p in model.parameters()),
        "layers": layer_data,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved layer RMS diagnostic to {output_path}")

    # Print table
    print(f"\n{'Layer':<14} {'Type':<22} {'RMS Mean':>10} {'RMS Min':>10} {'RMS Max':>10} {'RMS Std':>10}")
    print("-" * 78)
    for d in layer_data:
        lname = str(d["layer"])
        ltype = d["type"]
        rmean = d["hidden_rms_mean"]
        rmin = d.get("hidden_rms_min", "")
        rmax = d.get("hidden_rms_max", "")
        rstd = d.get("hidden_rms_std", "")
        print(f"{lname:<14} {ltype:<22} {rmean:>10.4f} {str(rmin):>10} {str(rmax):>10} {str(rstd):>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()
    diagnose(args.config, args.output, seed=args.seed, seq_len=args.seq_len)
