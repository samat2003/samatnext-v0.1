# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM

def benchmark_vram_for_model(model_class, config, seq_lengths, device):
    results = {}
    
    # Instantiate model
    model = model_class(config).to(device)
    
    for seq_len in seq_lengths:
        # Reset peak memory stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            
        # Create random input ids
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        
        # Forward pass
        try:
            # We enable grad to measure training VRAM (worst case)
            logits, _ = model(input_ids)
            # Dummy loss and backward pass
            loss = logits.sum()
            loss.backward()
            
            if device.type == "cuda":
                peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) # in MB
            else:
                peak_vram = 0.0 # CPU has no VRAM
            results[str(seq_len)] = f"{peak_vram:.2f} MB"
        except Exception as e:
            results[str(seq_len)] = f"OOM / Error ({type(e).__name__})"
            
    # Clean up model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running VRAM Benchmark on device: {device}")
    
    # Sequence lengths to profile
    seq_lengths = [128, 512, 1024, 2048]
    
    # Load configs
    samat_config_path = os.path.join(ROOT, "configs", "samatnext_350m.json")
    trans_config_path = os.path.join(ROOT, "configs", "transformer_350m_matched.json")
    
    samat_config = SamatNextConfig.from_json(samat_config_path)
    trans_config = TransformerConfig.from_json(trans_config_path)
    
    # Run profiling
    samat_results = benchmark_vram_for_model(SamatNextForCausalLM, samat_config, seq_lengths, device)
    trans_results = benchmark_vram_for_model(TransformerForCausalLM, trans_config, seq_lengths, device)
    
    output_data = {
        "samatnext": samat_results,
        "transformer": trans_results,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    }
    
    # Save JSON results
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    json_path = os.path.join(ROOT, "results", "tables", "vram_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved VRAM benchmark JSON report to {json_path}")
    
    # Save Markdown report
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    report_md_path = os.path.join(ROOT, "reports", "vram_benchmark.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# VRAM Benchmark Report\n\n")
        f.write(f"**Device:** `{device}` | **GPU Name:** `{output_data['gpu_name']}`\n\n")
        f.write("| Model Type | 128 context | 512 context | 1024 context | 2048 context |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **SamatNext-v0.1** | {samat_results['128']} | {samat_results['512']} | {samat_results['1024']} | {samat_results['2048']} |\n")
        f.write(f"| **Matched Transformer** | {trans_results['128']} | {trans_results['512']} | {trans_results['1024']} | {trans_results['2048']} |\n")
        
    print(f"Saved VRAM benchmark Markdown report to {report_md_path}")

if __name__ == "__main__":
    main()
