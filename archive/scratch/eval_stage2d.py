import os
import sys
import json
import re
import ast
import traceback
import torch
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_stage2a_fixed import generate_fixed

def check_syntax(code_str):
    try:
        ast.parse(code_str)
        return True, "Valid"
    except SyntaxError:
        return False, "SyntaxError"
    except Exception as e:
        return False, type(e).__name__

def check_execution_with_error(code_str, tests):
    try:
        local_scope = {}
        exec(code_str, {}, local_scope)
        for test in tests:
            exec(test, {}, local_scope)
        return True, "Pass"
    except Exception as e:
        return False, type(e).__name__

def check_function_name(code_str, expected_name):
    try:
        match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code_str)
        if match:
            return match.group(1) == expected_name
    except:
        pass
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    ckpt_path = args.ckpt or os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2d_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        sys.exit(1)

    print(f"Checkpoint : {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    eval_file = os.path.join(ROOT, "data", "stage2d_holdout_eval.jsonl")
    prompts = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    n = len(prompts)
    syntax_count = eos_stop_count = func_name_match_count = test_pass_count = 0
    failure_types = Counter()
    samples = []

    print(f"\nEvaluating {n} synthetic prompts...\n")
    for i, item in enumerate(prompts):
        instr = item["prompt"]
        expected_name = item["function_name"]
        tests = item["tests"]

        response, stopped = generate_fixed(model, tok, instr, device, max_new_tokens=128)
        clean_resp = response.replace(tok.eos_token, "").strip()

        valid, syn_err = check_syntax(clean_resp)
        func_match = check_function_name(clean_resp, expected_name)
        
        test_pass = False
        exec_err = "SyntaxError"
        if valid:
            test_pass, exec_err = check_execution_with_error(clean_resp, tests)
            
        if valid:   syntax_count  += 1
        if stopped: eos_stop_count += 1
        if func_match: func_name_match_count += 1
        if test_pass: 
            test_pass_count += 1
        else:
            err = exec_err if valid else syn_err
            failure_types[err] += 1

        res = {
            "instruction": instr,
            "expected_name": expected_name,
            "generated":   clean_resp,
            "syntax_valid": valid,
            "stopped_at_eos": stopped,
            "func_name_match": func_match,
            "test_pass": test_pass,
            "error": "None" if test_pass else (exec_err if valid else syn_err)
        }
        if i < 20:
            samples.append(res)

    name_error_count = failure_types.get("NameError", 0)
    # Wrong task is any failure that is not a SyntaxError and not a NameError (e.g. AssertionError, TypeError)
    wrong_task_count = sum(cnt for err, cnt in failure_types.items() if err not in ["SyntaxError", "NameError"] and err != "None")

    print("=== STAGE 2D EVALUATION ===")
    print(f"Checkpoint          : {ckpt_path}")
    print(f"Syntax Valid        : {syntax_count}/{n}  ({syntax_count/n*100:.1f}%)")
    print(f"Stopped at EOS      : {eos_stop_count}/{n}  ({eos_stop_count/n*100:.1f}%)")
    print(f"Func Name Match     : {func_name_match_count}/{n}  ({func_name_match_count/n*100:.1f}%)")
    print(f"Unit Test Pass Rate : {test_pass_count}/{n}  ({test_pass_count/n*100:.1f}%)")
    print(f"NameError Rate      : {name_error_count}/{n}  ({name_error_count/n*100:.1f}%)")
    print(f"Wrong-Task Rate     : {wrong_task_count}/{n}  ({wrong_task_count/n*100:.1f}%)")
    print("\nFailures Grouped By Type:")
    for err, cnt in failure_types.most_common():
        print(f"  {err}: {cnt}")
        
    print("\n--- 20 Sample Generations ---")
    for j, s in enumerate(samples, 1):
        print(f"\n[{j}] {s['instruction']}")
        print(f"    syntax={s['syntax_valid']}  func_match={s['func_name_match']}  "
              f"test_pass={s['test_pass']}  eos_stop={s['stopped_at_eos']}  err={s['error']}")
        print(f"    Output:\n{s['generated']}")

if __name__ == "__main__":
    main()
