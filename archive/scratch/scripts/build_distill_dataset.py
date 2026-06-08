import os
import sys
import json
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets import load_dataset

def is_bad_target(out):
    if not out or not out.strip():
        return True
    out_lower = out.lower()
    bad_phrases = [
        "i need more details",
        "i need more information",
        "please provide",
        "could you please describe",
        "i cannot provide",
        "i can't provide",
        "as an ai"
    ]
    for p in bad_phrases:
        if p in out_lower:
            return True
    return False

def main():
    print("Loading datasets...")
    random.seed(42)
    
    final_data = []
    seen_prompts = set()
    empty_prompt_count = 0
    empty_target_count = 0
    duplicate_prompt_count = 0
    bad_phrase_count = 0
    
    source_counts = {
        "Muennighoff/mbpp": 0,
        "iamtarun/python_code_instructions_18k_alpaca": 0,
        "sahil2801/CodeAlpaca-20k": 0
    }

    def process_and_add(record):
        nonlocal empty_prompt_count, empty_target_count, duplicate_prompt_count, bad_phrase_count
        prompt = record["prompt"].strip()
        target = record["target"].strip()
        
        if not prompt:
            empty_prompt_count += 1
            return False
        if not target:
            empty_target_count += 1
            return False
        if prompt in seen_prompts:
            duplicate_prompt_count += 1
            return False
        if is_bad_target(target):
            bad_phrase_count += 1
            return False
            
        seen_prompts.add(prompt)
        final_data.append(record)
        source_counts[record["source"]] += 1
        return True

    # 1. MBPP (need 500)
    try:
        ds_mbpp = load_dataset("google-research-datasets/mbpp", split="train+validation+test")
        mbpp_list = list(ds_mbpp)
        random.shuffle(mbpp_list)
        for ex in mbpp_list:
            if source_counts["Muennighoff/mbpp"] >= 500:
                break
            record = {
                "id": f"mbpp_{ex['task_id']}",
                "source": "Muennighoff/mbpp",
                "type": "generation",
                "prompt": "Write a Python function to " + ex['text'],
                "target": ex['code']
            }
            process_and_add(record)
    except Exception as e:
        print(f"Error loading MBPP: {e}")

    # 2. 18k Alpaca (need 300)
    try:
        ds_18k = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        alpaca18k_list = list(ds_18k)
        random.shuffle(alpaca18k_list)
        for i, ex in enumerate(alpaca18k_list):
            if source_counts["iamtarun/python_code_instructions_18k_alpaca"] >= 300:
                break
            p = ex['instruction']
            if ex.get('input'):
                p += "\n" + ex['input']
            record = {
                "id": f"18kalpaca_{i}",
                "source": "iamtarun/python_code_instructions_18k_alpaca",
                "type": "generation",
                "prompt": p,
                "target": ex['output']
            }
            process_and_add(record)
    except Exception as e:
        print(f"Error loading 18k alpaca: {e}")

    # 3. CodeAlpaca-20k (need 200)
    try:
        ds_20k = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        codealpaca_list = list(ds_20k)
        random.shuffle(codealpaca_list)
        for i, ex in enumerate(codealpaca_list):
            if source_counts["sahil2801/CodeAlpaca-20k"] >= 200:
                break
            if "python" in ex['instruction'].lower() or "python" in ex.get('input', '').lower():
                p = ex['instruction']
                if ex.get('input'):
                    p += "\n" + ex['input']
                record = {
                    "id": f"codealpaca_{i}",
                    "source": "sahil2801/CodeAlpaca-20k",
                    "type": "generation",
                    "prompt": p,
                    "target": ex['output']
                }
                process_and_add(record)
    except Exception as e:
        print(f"Error loading CodeAlpaca-20k: {e}")

    out_path = os.path.join(ROOT, "data", "stage2_distill_dataset.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final_data:
            f.write(json.dumps(r) + "\n")
            
    print("\n--- DATASET GENERATION REPORT ---")
    print(f"Total examples saved: {len(final_data)}")
    print("Count by source:")
    for k, v in source_counts.items():
        print(f"  {k}: {v}")
    print(f"Empty prompt count: {empty_prompt_count}")
    print(f"Empty target count: {empty_target_count}")
    print(f"Duplicate prompt count: {duplicate_prompt_count}")
    print(f"Examples containing 'I need more details' / bad phrases: {bad_phrase_count}")
    
    print(f"\nSaved to {out_path}")
    
    print("\n--- FIRST 10 EXAMPLES ---")
    for i in range(min(10, len(final_data))):
        print(f"\n--- Example {i+1} ---")
        print(json.dumps(final_data[i], indent=2))

if __name__ == "__main__":
    main()
