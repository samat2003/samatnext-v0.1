# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import time
import re
import argparse
import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM
from scripts.eval_execute import run_test_subprocess

@torch.no_grad()
def greedy_generate(model, tokenizer, device, prompt_str, max_new_tokens=256):
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
    generated = []
    
    # Qwen token IDs
    eos_id = tokenizer.eos_token_id
    # <|im_end|> token ID is 151645
    im_end_id = 151645
    
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_token = torch.argmax(logits[0, -1, :]).item()
        
        if next_token == eos_id or next_token == im_end_id:
            break
            
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        if input_ids.shape[1] > 2048:
            input_ids = input_ids[:, -2048:]
            
    return tokenizer.decode(generated, skip_special_tokens=False)

def extract_python_code(text):
    # Strip markdown fences if present
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["samatnext", "transformer"])
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, required=True, choices=["humaneval", "mbpp", "custom_retention"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load model
    print(f"Loading config from {args.model_config}...")
    if args.model == "samatnext":
        config = SamatNextConfig.from_json(args.model_config)
        model = SamatNextForCausalLM(config).to(device)
    else:
        config = TransformerConfig.from_json(args.model_config)
        model = TransformerForCausalLM(config).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    important_missing = [k for k in missing_keys if "freqs_cis" not in k]
    important_unexpected = [k for k in unexpected_keys if "freqs_cis" not in k]
    if important_missing:
        raise RuntimeError(f"Missing key(s) in state_dict: {important_missing}")
    if important_unexpected:
        raise RuntimeError(f"Unexpected key(s) in state_dict: {important_unexpected}")
    model.eval()

    # Load dataset
    dataset_items = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            dataset_items.append(json.loads(line))
            
    print(f"Loaded {len(dataset_items)} items of type {args.dataset_type}.")

    raw_generations_path = os.path.join(args.output_dir, "raw_generations.jsonl")
    execution_errors_path = os.path.join(args.output_dir, "execution_errors.jsonl")
    
    raw_gen_file = open(raw_generations_path, "w", encoding="utf-8")
    exec_err_file = open(execution_errors_path, "w", encoding="utf-8")

    passed_count = 0
    total_count = len(dataset_items)
    category_counts = {}
    timeout_count = 0

    for idx, item in enumerate(dataset_items):
        task_id = item.get("task_id", item.get("id", f"task_{idx}"))
        prompt_code = item.get("prompt", "")
        tests = item.get("test_list", item.get("tests", []))
        if isinstance(tests, str):
            tests = [tests]
        # Append HumanEval standard test cases
        if "test" in item and item["test"]:
            tests.append(item["test"])
            
        entry_point = item.get("entry_point", item.get("function_name", "NONE"))

        # Build Qwen chat prompt
        chat_prompt = (
            f"<|im_start|>user\n"
            f"Complete the following Python function. Return ONLY the code, no explanation.\n\n"
            f"```python\n{prompt_code}\n```\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"```python\n{prompt_code}"
        )

        # Generate response
        raw_response = greedy_generate(model, tok, device, chat_prompt, max_new_tokens=args.max_new_tokens)
        full_code = prompt_code + "\n" + raw_response
        clean_code = extract_python_code(full_code)

        # Run test execution
        success, err, is_timeout = run_test_subprocess(clean_code, entry_point, tests, timeout_seconds=args.timeout_seconds)

        status_str = "PASS" if success else "FAIL"
        if success:
            passed_count += 1
        if is_timeout:
            timeout_count += 1
            err = "Timeout"

        # Log to files
        raw_gen_file.write(json.dumps({
            "task_id": task_id,
            "prompt": prompt_code,
            "generation": raw_response,
            "clean_code": clean_code
        }) + "\n")
        raw_gen_file.flush()
        
        if not success:
            exec_err_file.write(json.dumps({
                "task_id": task_id,
                "error": err,
                "timeout": is_timeout
            }) + "\n")
            exec_err_file.flush()

        # Track categories
        cat = err if err else "PASS"
        category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"[{idx+1}/{total_count}] {task_id:<28} {status_str} {f'({err})' if err else ''}")

        # Clear CUDA cache to prevent VRAM memory accumulation and paging
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_gen_file.close()
    exec_err_file.close()

    pass_rate = passed_count / total_count if total_count > 0 else 0.0
    print(f"\nFinal Result: {passed_count}/{total_count} Passed ({pass_rate:.1%})")
    print(f"Total Timeouts: {timeout_count}")

    # Write Category Breakdown
    with open(os.path.join(args.output_dir, "category_breakdown.json"), "w") as f:
        json.dump(category_counts, f, indent=4)

    # Save specific benchmark JSON results
    result_filename = f"{args.dataset_type}_results.json"
    results_path = os.path.join(args.output_dir, result_filename)
    with open(results_path, "w") as f:
        json.dump({
            "dataset_type": args.dataset_type,
            "pass_rate": pass_rate,
            "passed": passed_count,
            "total": total_count,
            "timeouts": timeout_count,
            "category_breakdown": category_counts
        }, f, indent=4)

    # Save Evaluation Manifest
    eval_manifest = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "dataset_path": args.dataset,
        "dataset_type": args.dataset_type,
        "timeout_seconds": args.timeout_seconds,
        "pass_rate": pass_rate,
        "passed_count": passed_count,
        "total_count": total_count,
        "timeout_count": timeout_count
    }
    with open(os.path.join(args.output_dir, "eval_manifest.json"), "w") as f:
        json.dump(eval_manifest, f, indent=4)

if __name__ == "__main__":
    main()
