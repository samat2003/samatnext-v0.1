# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import random
import hashlib
import ast
import argparse
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def get_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clean_code(text):
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```python") or lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def check_contamination_simple(prompt, target, humaneval_keys):
    # Check if prompt or target contains functions or prompts in HumanEval
    # Normalize strings
    p_norm = prompt.strip().lower()
    for key in humaneval_keys:
        if key in p_norm:
            return True
    return False

def load_humaneval_keys():
    keys = set()
    he_path = os.path.join(ROOT, "data", "HumanEval.jsonl")
    if os.path.exists(he_path):
        with open(he_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                keys.add(ex["entry_point"].strip().lower())
    return keys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Generate a tiny dataset quickly")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic shuffling")
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "data", "processed"))
    parser.add_argument("--write-manifest", type=str, default=os.path.join(ROOT, "data", "manifests", "data_manifest.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.write_manifest), exist_ok=True)

    print("Loading HumanEval entry points to filter contamination...")
    he_keys = load_humaneval_keys()
    print(f"Loaded {len(he_keys)} HumanEval entry points.")

    # Load Qwen tokenizer for token count estimation
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    # Define raw file sources and mapping
    raw_sources = {
        "stage2a_code_pretrain": os.path.join(ROOT, "data", "stage2a_code_pretrain.jsonl"),
        "stage3_paraphrase_train": os.path.join(ROOT, "data", "stage3_paraphrase_train.jsonl"),
        "stage5_teacher_distill": os.path.join(ROOT, "data", "stage5_teacher_distill.jsonl")
    }

    processed_files_info = {}
    removed_counts = {"empty": 0, "malformed": 0, "non_python": 0, "duplicate": 0, "contaminated": 0}

    for stage_name, raw_path in raw_sources.items():
        if not os.path.exists(raw_path):
            print(f"Warning: Raw source {raw_path} not found. Skipping.")
            continue
            
        print(f"Processing {stage_name}...")
        raw_data = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    raw_data.append(json.loads(line))
                except Exception:
                    removed_counts["malformed"] += 1

        cleaned_data = []
        seen_hashes = set()

        for idx, ex in enumerate(raw_data):
            # 1. Check empty
            prompt = ex.get("prompt", "")
            target = ex.get("target", ex.get("teacher_target", ""))
            
            if not prompt or not target:
                removed_counts["empty"] += 1
                continue
                
            prompt = prompt.strip()
            target = clean_code(target)
            
            # 2. Check duplicate
            ex_hash = hashlib.sha256(f"{prompt} || {target}".encode("utf-8")).hexdigest()
            if ex_hash in seen_hashes:
                removed_counts["duplicate"] += 1
                continue
            seen_hashes.add(ex_hash)
            
            # 3. Check contamination
            if check_contamination_simple(prompt, target, he_keys):
                removed_counts["contaminated"] += 1
                continue
                
            # 4. Check AST parsing
            # Combine prompt and target to see if it makes a valid Python program (or check target alone)
            if not is_valid_python(target):
                removed_counts["non_python"] += 1
                continue

            cleaned_data.append({
                "prompt": prompt,
                "target": target
            })

        # Shuffle deterministically
        random.shuffle(cleaned_data)

        # Handle smoke sizing
        if args.smoke:
            cleaned_data = cleaned_data[:50]

        # Save processed file
        out_filename = f"{stage_name}.jsonl" if stage_name != "stage5_teacher_distill" else "stage5_teacher_sft.jsonl"
        out_path = os.path.join(args.output_dir, out_filename)
        
        total_tokens = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in cleaned_data:
                f.write(json.dumps(ex) + "\n")
                if tok:
                    total_tokens += len(tok.encode(ex["prompt"] + ex["target"]))
                else:
                    total_tokens += len((ex["prompt"] + ex["target"]).split())

        sha256 = get_file_sha256(out_path)
        processed_files_info[out_filename] = {
            "path": out_path,
            "sha256": sha256,
            "sample_count": len(cleaned_data),
            "estimated_tokens": total_tokens
        }
        print(f"Saved {len(cleaned_data)} examples to {out_path} (SHA256: {sha256[:10]}...)")

    # Generate train/val/test splits from stage5 data or standard combined split
    # For baseline training, train.jsonl and val.jsonl are required.
    # We will generate train.jsonl and val.jsonl by splitting stage5_teacher_sft.jsonl.
    sft_info = processed_files_info.get("stage5_teacher_sft.jsonl")
    if sft_info:
        sft_path = sft_info["path"]
        sft_data = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                sft_data.append(json.loads(line))
        
        # Split 80% train, 10% val, 10% test
        n = len(sft_data)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_data = sft_data[:n_train]
        val_data = sft_data[n_train:n_train+n_val]
        test_data = sft_data[n_train+n_val:]
        
        splits = {
            "train.jsonl": train_data,
            "val.jsonl": val_data,
            "internal_test.jsonl": test_data
        }
        
        for name, split_data in splits.items():
            split_path = os.path.join(args.output_dir, name)
            total_tokens = 0
            with open(split_path, "w", encoding="utf-8") as f:
                for ex in split_data:
                    f.write(json.dumps(ex) + "\n")
                    if tok:
                        total_tokens += len(tok.encode(ex["prompt"] + ex["target"]))
                    else:
                        total_tokens += len((ex["prompt"] + ex["target"]).split())
                        
            sha256 = get_file_sha256(split_path)
            processed_files_info[name] = {
                "path": split_path,
                "sha256": sha256,
                "sample_count": len(split_data),
                "estimated_tokens": total_tokens
            }
            print(f"Saved split {name} with {len(split_data)} examples to {split_path}")

    # Set up benchmark pointers
    # Download MBPP from HF programmatically
    print("Preparing MBPP evaluation benchmark data...")
    mbpp_eval_path = os.path.join(ROOT, "data", "benchmark", "mbpp_eval.jsonl")
    os.makedirs(os.path.dirname(mbpp_eval_path), exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("Downloading MBPP dataset via Hugging Face datasets...")
        mbpp = load_dataset("google-research-datasets/mbpp")
        # Use 'test' split for MBPP held-out evaluation
        test_split = mbpp["test"]
        mbpp_eval_data = []
        for row in test_split:
            mbpp_eval_data.append({
                "task_id": row["task_id"],
                "prompt": row["text"],
                "code": row["code"],
                "test_imports": row.get("test_imports", []),
                "test_list": row.get("test_list", [])
            })
            
        if args.smoke:
            mbpp_eval_data = mbpp_eval_data[:10]
            
        with open(mbpp_eval_path, "w", encoding="utf-8") as f:
            for ex in mbpp_eval_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Successfully saved {len(mbpp_eval_data)} MBPP test problems to {mbpp_eval_path}")
    except Exception as e:
        print(f"Hugging Face MBPP load skipped or failed: {e}. Writing pointer.")
        # Write metadata pointer
        with open(mbpp_eval_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"info": "MBPP test set pointer", "status": "failed_load"}) + "\n")

    # Set up HumanEval pointer
    humaneval_pointer_path = os.path.join(ROOT, "data", "benchmark", "humaneval.jsonl")
    with open(humaneval_pointer_path, "w", encoding="utf-8") as f:
        # Load local HumanEval
        he_raw_path = os.path.join(ROOT, "data", "HumanEval.jsonl")
        if os.path.exists(he_raw_path):
            with open(he_raw_path, "r", encoding="utf-8") as he_f:
                for line in he_f:
                    f.write(line)
            print(f"Copied HumanEval data to {humaneval_pointer_path}")
        else:
            f.write(json.dumps({"info": "HumanEval pointer"}) + "\n")

    # Custom retention eval
    custom_ret_path = os.path.join(ROOT, "data", "benchmark", "custom_retention_eval.jsonl")
    custom_raw_path = os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")
    with open(custom_ret_path, "w", encoding="utf-8") as f:
        if os.path.exists(custom_raw_path):
            with open(custom_raw_path, "r", encoding="utf-8") as cr_f:
                for line in cr_f:
                    f.write(line)
            print(f"Copied custom retention evaluation data to {custom_ret_path}")
        else:
            f.write(json.dumps({"info": "Custom retention pointer"}) + "\n")

    # Write manifest JSON
    manifest_data = {
        "seed": args.seed,
        "smoke": args.smoke,
        "removed_counts": removed_counts,
        "processed_files": processed_files_info,
        "benchmarks": {
            "humaneval": {
                "path": humaneval_pointer_path,
                "sha256": get_file_sha256(humaneval_pointer_path) if os.path.exists(humaneval_pointer_path) else None
            },
            "mbpp": {
                "path": mbpp_eval_path,
                "sha256": get_file_sha256(mbpp_eval_path) if os.path.exists(mbpp_eval_path) else None
            },
            "custom_retention": {
                "path": custom_ret_path,
                "sha256": get_file_sha256(custom_ret_path) if os.path.exists(custom_ret_path) else None
            }
        }
    }
    
    with open(args.write_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    print(f"Saved data manifest to {args.write_manifest}")

    # Write SHA256 hashes text file
    hashes_path = os.path.join(ROOT, "data", "manifests", "data_hashes.txt")
    with open(hashes_path, "w", encoding="utf-8") as f:
        for name, info in processed_files_info.items():
            f.write(f"{info['sha256']}  data/processed/{name}\n")
    print(f"Saved data hashes list to {hashes_path}")

if __name__ == "__main__":
    main()
