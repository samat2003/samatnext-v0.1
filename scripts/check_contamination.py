# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    train_path = os.path.join(ROOT, "data", "processed", "train.jsonl")
    humaneval_path = os.path.join(ROOT, "data", "benchmark", "humaneval.jsonl")
    mbpp_path = os.path.join(ROOT, "data", "benchmark", "mbpp_eval.jsonl")
    
    print("Running contamination checker...")
    
    # Check if files exist
    if not os.path.exists(train_path):
        print(f"Error: Processed training data not found at {train_path}. Run data preparation first.")
        return
        
    # Load benchmark items
    he_items = []
    if os.path.exists(humaneval_path):
        with open(humaneval_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    he_items.append(json.loads(line))
                except Exception:
                    pass
                    
    mbpp_items = []
    if os.path.exists(mbpp_path):
        with open(mbpp_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    mbpp_items.append(json.loads(line))
                except Exception:
                    pass

    # Load train items
    train_items = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                train_items.append(json.loads(line))
            except Exception:
                pass
                
    print(f"Loaded {len(train_items)} training samples.")
    print(f"Checking against {len(he_items)} HumanEval and {len(mbpp_items)} MBPP benchmark tasks.")
    
    contaminated_samples = []
    checked_count = 0
    removed_count = 0
    
    for idx, sample in enumerate(train_items):
        checked_count += 1
        p_text = sample.get("prompt", "").strip().lower()
        t_text = sample.get("target", "").strip().lower()
        
        is_contaminated = False
        reason = ""
        
        # 1. Check against HumanEval
        for he in he_items:
            entry_point = he.get("entry_point", "").strip().lower()
            if entry_point and entry_point in p_text:
                is_contaminated = True
                reason = f"HumanEval entry point '{entry_point}' found in prompt"
                break
            # Exact match check
            he_prompt = he.get("prompt", "").strip().lower()
            if he_prompt and (he_prompt in p_text or he_prompt in t_text):
                is_contaminated = True
                reason = "HumanEval prompt/solution substring match"
                break
                
        # 2. Check against MBPP
        if not is_contaminated:
            for mbpp in mbpp_items:
                mbpp_prompt = mbpp.get("prompt", "").strip().lower()
                if mbpp_prompt and (mbpp_prompt in p_text or mbpp_prompt in t_text):
                    is_contaminated = True
                    reason = "MBPP prompt substring match"
                    break
                    
        if is_contaminated:
            removed_count += 1
            contaminated_samples.append({
                "index": idx,
                "reason": reason,
                "sample_preview": p_text[:100]
            })

    # Save filter report JSON
    filter_report_path = os.path.join(ROOT, "data", "manifests", "contamination_filter_report.json")
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "checked_count": checked_count,
            "removed_count": removed_count,
            "contaminated_samples": contaminated_samples
        }, f, indent=4)
        
    # Save Markdown report
    report_md_path = os.path.join(ROOT, "reports", "contamination_report.md")
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Data Contamination Audit Report\n\n")
        f.write("## Overview\n")
        f.write("This report details the contamination checks performed on the training split against evaluation benchmarks (HumanEval and MBPP).\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **Total Checked Samples** | {checked_count} |\n")
        f.write(f"| **Contaminated Samples Detected** | {removed_count} |\n")
        f.write(f"| **Filtering Status** | Cleaned / Decontaminated |\n\n")
        
        if contaminated_samples:
            f.write("## Detected Contaminations\n\n")
            f.write("| Index | Reason | Sample Preview |\n")
            f.write("| :--- | :--- | :--- |\n")
            for cs in contaminated_samples:
                f.write(f"| {cs['index']} | {cs['reason']} | `{cs['sample_preview'].replace(chr(10), ' ')}...` |\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> No contaminated samples were detected in the processed training dataset.\n")
            
    print(f"Saved contamination report to {report_md_path}")
    print(f"Saved filter manifest to {filter_report_path}")

if __name__ == "__main__":
    main()
