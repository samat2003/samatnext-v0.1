# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM

def count_model_parameters(model, model_type):
    # Initialize counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    embedding_params = 0
    lm_head_params = 0
    attention_params = 0
    mixer_params = 0
    mlp_params = 0
    norm_params = 0
    verifier_head_params = 0
    
    # Check if embedding and lm_head are tied
    if model_type == "samatnext":
        embed_weight = model.model.embed_tokens.weight
        lm_head_weight = model.lm_head.weight
        is_tied = embed_weight is lm_head_weight
        
        # Breakdown parameters
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                embedding_params += param.numel()
            elif "lm_head" in name:
                lm_head_params += param.numel()
            elif "verifier_head" in name:
                verifier_head_params += param.numel()
            elif "layernorm" in name or "norm" in name:
                norm_params += param.numel()
            elif "mlp" in name:
                mlp_params += param.numel()
            elif "mixer" in name:
                mixer_params += param.numel()
                # Differential Attention counts as attention, linear-state mixer counts as mixer
                if "differential_attention" in name or "q_proj_diff" in name or "k_proj_diff" in name or "alpha" in name:
                    attention_params += param.numel()
    else:
        embed_weight = model.embed_tokens.weight
        lm_head_weight = model.lm_head.weight
        is_tied = embed_weight is lm_head_weight
        
        # Breakdown parameters
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                embedding_params += param.numel()
            elif "lm_head" in name:
                lm_head_params += param.numel()
            elif "layernorm" in name or "norm" in name:
                norm_params += param.numel()
            elif "mlp" in name:
                mlp_params += param.numel()
            elif "attn" in name:
                attention_params += param.numel()

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "embedding_parameters": embedding_params,
        "lm_head_parameters": lm_head_params,
        "attention_parameters": attention_params,
        "mixer_parameters": mixer_params,
        "mlp_parameters": mlp_params,
        "norm_parameters": norm_params,
        "verifier_head_parameters": verifier_head_params,
        "embeddings_tied": is_tied
    }

def main():
    # Load tokenizer length
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    tokenizer_len = len(tok)
    
    # Load configs
    samat_config_path = os.path.join(ROOT, "configs", "samatnext_350m.json")
    trans_config_path = os.path.join(ROOT, "configs", "transformer_350m_matched.json")
    
    samat_config = SamatNextConfig.from_json(samat_config_path)
    trans_config = TransformerConfig.from_json(trans_config_path)
    
    # Instantiate models
    samat_model = SamatNextForCausalLM(samat_config)
    trans_model = TransformerForCausalLM(trans_config)
    
    # Count parameters
    samat_stats = count_model_parameters(samat_model, "samatnext")
    trans_stats = count_model_parameters(trans_model, "transformer")
    
    # Attach config specs
    samat_report = {
        **samat_stats,
        "vocab_size": samat_config.vocab_size,
        "tokenizer_length": tokenizer_len,
        "context_length": samat_config.max_position_embeddings,
        "hidden_size": samat_config.hidden_size,
        "layer_count": samat_config.num_layers,
        "ffn_intermediate_size": samat_config.intermediate_size,
        "num_heads": samat_config.num_attention_heads,
        "kv_heads": samat_config.num_key_value_heads,
        "mixer_pattern": samat_config.mixer_pattern,
        "use_rope": samat_config.use_rope,
    }
    
    trans_report = {
        **trans_stats,
        "vocab_size": trans_config.vocab_size,
        "tokenizer_length": tokenizer_len,
        "context_length": trans_config.max_position_embeddings,
        "hidden_size": trans_config.hidden_size,
        "layer_count": trans_config.num_layers,
        "ffn_intermediate_size": trans_config.intermediate_size,
        "num_heads": trans_config.num_attention_heads,
        "kv_heads": trans_config.num_key_value_heads,
        "use_rope": trans_config.use_rope,
    }
    
    output_data = {
        "samatnext": samat_report,
        "transformer": trans_report
    }
    
    # Ensure directory exists
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    
    # Save to JSON
    json_path = os.path.join(ROOT, "results", "tables", "parameter_counts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved parameter count JSON report to {json_path}")
    
    # Save to Markdown
    md_path = os.path.join(ROOT, "results", "tables", "parameter_counts.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Parameter Count and Structural Specifications\n\n")
        f.write("| Specification / Parameter Type | SamatNext-v0.1 | Matched Transformer Baseline |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Total Parameters** | {samat_report['total_parameters']:,} | {trans_report['total_parameters']:,} |\n")
        f.write(f"| **Trainable Parameters** | {samat_report['trainable_parameters']:,} | {trans_report['trainable_parameters']:,} |\n")
        f.write(f"| **Embedding Parameters** | {samat_report['embedding_parameters']:,} | {trans_report['embedding_parameters']:,} |\n")
        f.write(f"| **LM Head Parameters** | {samat_report['lm_head_parameters']:,} | {trans_report['lm_head_parameters']:,} |\n")
        f.write(f"| **Attention Parameters** | {samat_report['attention_parameters']:,} | {trans_report['attention_parameters']:,} |\n")
        f.write(f"| **Mixer Parameters (Non-Attn)** | {samat_report['mixer_parameters']:,} | N/A (MHA only) |\n")
        f.write(f"| **MLP (FFN) Parameters** | {samat_report['mlp_parameters']:,} | {trans_report['mlp_parameters']:,} |\n")
        f.write(f"| **Normalization Parameters** | {samat_report['norm_parameters']:,} | {trans_report['norm_parameters']:,} |\n")
        f.write(f"| **Verifier Head Parameters** | {samat_report['verifier_head_parameters']:,} | N/A |\n")
        f.write(f"| **Embeddings Tied Status** | {'Tied' if samat_report['embeddings_tied'] else 'Untied'} | {'Tied' if trans_report['embeddings_tied'] else 'Untied'} |\n")
        f.write(f"| **Vocab Size** | {samat_report['vocab_size']:,} | {trans_report['vocab_size']:,} |\n")
        f.write(f"| **Tokenizer Length** | {samat_report['tokenizer_length']:,} | {trans_report['tokenizer_length']:,} |\n")
        f.write(f"| **Context Length** | {samat_report['context_length']:,} | {trans_report['context_length']:,} |\n")
        f.write(f"| **Hidden Size** | {samat_report['hidden_size']:,} | {trans_report['hidden_size']:,} |\n")
        f.write(f"| **Layer Count** | {samat_report['layer_count']} | {trans_report['layer_count']} |\n")
        f.write(f"| **FFN Intermediate Size** | {samat_report['ffn_intermediate_size']:,} | {trans_report['ffn_intermediate_size']:,} |\n")
        f.write(f"| **Number of Heads** | {samat_report['num_heads']} | {trans_report['num_heads']} |\n")
        f.write(f"| **KV Heads** | {samat_report['kv_heads']} | {trans_report['kv_heads']} |\n")
        f.write(f"| **Mixer Pattern / Type** | {samat_report['mixer_pattern']} | Multi-Head Attention |\n")
        f.write(f"| **RoPE Status** | {'Enabled' if samat_report['use_rope'] else 'Disabled'} | {'Enabled' if trans_report['use_rope'] else 'Disabled'} |\n")
        
    print(f"Saved parameter count Markdown report to {md_path}")

if __name__ == "__main__":
    main()
