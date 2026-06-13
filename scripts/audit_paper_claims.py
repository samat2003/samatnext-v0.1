# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def audit_parameters():
    print("Auditing parameter counts...")
    json_path = os.path.join(ROOT, "results", "tables", "parameter_counts.json")
    if not os.path.exists(json_path):
        print("FAIL: parameter_counts.json not found")
        return "FAIL"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    samat = data.get("samatnext", {})
    trans = data.get("transformer", {})
    
    # Audit values
    expected = {
        "samat_total": 356082440,
        "trans_total": 356082432,
        "samat_attn": 9437192,
        "samat_mixer": 47185928,
        "samat_mlp": 75497472,
        "trans_attn": 37748736,
        "trans_mlp": 84934656,
        "norm": 25344,
        "vocab_size": 151936,
        "context_length": 8192,
        "hidden_size": 768,
        "layer_count": 16,
        "samat_ffn": 2048,
        "trans_ffn": 2304,
        "samat_kv_heads": 4,
        "trans_kv_heads": 12,
        "embeddings_tied": False
    }
    
    checks = {
        "samat_total": samat.get("total_parameters") == expected["samat_total"],
        "trans_total": trans.get("total_parameters") == expected["trans_total"],
        "samat_attn": samat.get("attention_parameters") == expected["samat_attn"],
        "samat_mixer": samat.get("mixer_parameters") == expected["samat_mixer"],
        "samat_mlp": samat.get("mlp_parameters") == expected["samat_mlp"],
        "trans_attn": trans.get("attention_parameters") == expected["trans_attn"],
        "trans_mlp": trans.get("mlp_parameters") == expected["trans_mlp"],
        "norm": samat.get("norm_parameters") == expected["norm"] and trans.get("norm_parameters") == expected["norm"],
        "vocab_size": samat.get("vocab_size") == expected["vocab_size"] and trans.get("vocab_size") == expected["vocab_size"],
        "context_length": samat.get("context_length") == expected["context_length"] and trans.get("context_length") == expected["context_length"],
        "hidden_size": samat.get("hidden_size") == expected["hidden_size"] and trans.get("hidden_size") == expected["hidden_size"],
        "layer_count": samat.get("layer_count") == expected["layer_count"] and trans.get("layer_count") == expected["layer_count"],
        "samat_ffn": samat.get("ffn_intermediate_size") == expected["samat_ffn"],
        "trans_ffn": trans.get("ffn_intermediate_size") == expected["trans_ffn"],
        "samat_kv_heads": samat.get("kv_heads") == expected["samat_kv_heads"],
        "trans_kv_heads": trans.get("kv_heads") == expected["trans_kv_heads"],
        "embeddings_tied": samat.get("embeddings_tied") == expected["embeddings_tied"] and trans.get("embeddings_tied") == expected["embeddings_tied"]
    }
    
    for k, v in checks.items():
        if not v:
            print(f"FAIL: parameter check '{k}' failed (expected {expected[k]})")
            return "FAIL"
            
    print("PASS: Parameter counts and model configuration audited successfully")
    return "PASS"

def audit_layer_pattern():
    print("Auditing layer alternation pattern...")
    from models.samat_next.config import SamatNextConfig
    from models.samat_next.layers import SamatNextBlock
    
    config_path = os.path.join(ROOT, "configs", "samatnext_350m.json")
    if not os.path.exists(config_path):
        print("FAIL: samatnext_350m.json config not found")
        return "FAIL"
        
    config = SamatNextConfig.from_json(config_path)
    
    # Check layer alternation pattern in SamatNextBlock initialization
    for layer_idx in range(config.num_layers):
        block = SamatNextBlock(config, layer_idx)
        mixer_class = block.mixer.__class__.__name__
        if layer_idx % 2 == 0:
            if mixer_class != "DeltaNetInspiredLinearStateMixer":
                print(f"FAIL: Layer {layer_idx} (even) should be DeltaNetInspiredLinearStateMixer, got {mixer_class}")
                return "FAIL"
        else:
            if mixer_class != "DifferentialAttention":
                print(f"FAIL: Layer {layer_idx} (odd) should be DifferentialAttention, got {mixer_class}")
                return "FAIL"
                
    print("PASS: Alternation pattern verified (Even layers: DeltaNet, Odd layers: DifferentialAttention)")
    return "PASS"

def audit_results_table():
    print("Auditing results table consistency...")
    main_tex_path = os.path.join(ROOT, "paper", "main.tex")
    if not os.path.exists(main_tex_path):
        print("FAIL: paper/main.tex not found")
        return "FAIL"
        
    content = open(main_tex_path, "r", encoding="utf-8").read()
    
    expected_rows = [
        r"Transformer & Scratch $\rightarrow$ Stage5 & 97.6\% & 0.8\% & 3.3\% \\",
        r"SamatNext & Scratch $\rightarrow$ Stage5 & 97.6\% & 0.8\% & 1.3\% \\",
        r"Transformer & Curriculum LR=3e-6 & 49.4\% & 4.0\% & 0.0\% \\",
        r"Transformer & Curriculum LR=1e-5 & 97.6\% & 6.0\% & 3.0\% \\",
        r"Transformer & Curriculum LR=3e-5 & 97.6\% & 3.2\% & 2.0\% \\",
        r"SamatNext & Curriculum LR=3e-6 & 83.0\% & 70.2\% & 4.3\% \\"
    ]
    
    for row in expected_rows:
        normalized_row = re.sub(r"\s+", "", row)
        normalized_content = re.sub(r"\s+", "", content)
        if normalized_row not in normalized_content:
            print(f"FAIL: results table missing row: {row}")
            return "FAIL"
            
    print("PASS: Results table values verified in main.tex")
    return "PASS"

def audit_vram_table():
    print("Auditing VRAM table provenance...")
    json_path = os.path.join(ROOT, "results", "tables", "vram_benchmark.json")
    if not os.path.exists(json_path):
        print("FAIL: vram_benchmark.json not found")
        return "FAIL"
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    main_tex_path = os.path.join(ROOT, "paper", "main.tex")
    if not os.path.exists(main_tex_path):
        print("FAIL: paper/main.tex not found")
        return "FAIL"
        
    content = open(main_tex_path, "r", encoding="utf-8").read()
    
    samat_vram = data.get("samatnext", {})
    trans_vram = data.get("transformer", {})
    
    for seq_len in ["128", "512", "1024", "2048"]:
        s_val = samat_vram.get(seq_len)
        t_val = trans_vram.get(seq_len)
        
        # Verify that these values exist in the table in main.tex
        pattern = re.escape(seq_len) + r"\s*&\s*" + re.escape(s_val) + r"\s*&\s*" + re.escape(t_val)
        if not re.search(pattern, content):
            print(f"FAIL: VRAM table missing row for seq_len {seq_len}: {s_val} vs {t_val}")
            return "FAIL"
            
    print("PASS: VRAM table values verified in main.tex")
    return "PASS"

def audit_artifact_commits():
    print("Auditing artifact manifest consistency...")
    source_commit = "41636fe45f01838dd10ccdf7cb94d75fa6061ae2"
    repro_commit = "525665fe790b18668251dad6698fe9bfe0ca27ca"
    
    main_tex_path = os.path.join(ROOT, "paper", "main.tex")
    paper_readme_path = os.path.join(ROOT, "paper", "README.md")
    
    for path in [main_tex_path, paper_readme_path]:
        if not os.path.exists(path):
            print(f"FAIL: File not found: {path}")
            return "FAIL"
        content = open(path, "r", encoding="utf-8").read()
        if source_commit not in content:
            print(f"FAIL: source commit {source_commit} not in {path}")
            return "FAIL"
        if repro_commit not in content:
            print(f"FAIL: reproducibility commit {repro_commit} not in {path}")
            return "FAIL"
            
    print("PASS: Artifact commits verified in both main.tex and paper/README.md")
    return "PASS"

def main():
    print("Running paper claims audit...\n" + "="*40)
    p_status = audit_parameters()
    l_status = audit_layer_pattern()
    r_status = audit_results_table()
    v_status = audit_vram_table()
    c_status = audit_artifact_commits()
    
    print("="*40)
    print(f"Parameter Config Audit: {p_status}")
    print(f"Layer Pattern Audit:    {l_status}")
    print(f"Results Table Audit:    {r_status}")
    print(f"VRAM Table Audit:       {v_status}")
    print(f"Artifact Commits Audit: {c_status}")
    
    if "FAIL" in [p_status, l_status, r_status, v_status, c_status]:
        print("\nAUDIT STATUS: FAIL")
        sys.exit(1)
    else:
        print("\nAUDIT STATUS: PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
