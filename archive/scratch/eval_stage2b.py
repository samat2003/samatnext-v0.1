import os
import sys
import json
import ast
import re
import torch
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_unseen import UNSEEN_PROMPTS, check_syntax, check_execution
from scripts.eval_stage2a_fixed import generate_fixed, check_repetition

def check_function_name(code_str, expected_name):
    try:
        # Find all function definitions
        match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code_str)
        if match:
            return match.group(1) == expected_name
    except:
        pass
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to evaluate (default: best then latest)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    assert tok.eos_token_id == 151645
    assert tok.pad_token_id == 151643

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    ckpt_candidates = [
        args.ckpt,
        os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2b_hq_best.pt"),
        os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2b_hq_latest.pt"),
    ]
    ckpt_path = next((c for c in ckpt_candidates if c and os.path.exists(c)), None)
    if ckpt_path is None:
        print("ERROR: No checkpoint found.")
        sys.exit(1)

    print(f"Checkpoint : {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    from scripts.eval_unseen import UNSEEN_PROMPTS_NATURAL

    def evaluate_set(prompt_set, set_name):
        n = len(prompt_set)
        syntax_count = rep_count = eos_stop_count = 0
        func_name_match_count = test_pass_count = 0
        samples = []

        print(f"\nEvaluating {n} {set_name} prompts...\n")
        for i, item in enumerate(prompt_set):
            instr = item["instruction"]
            test_code = item["test"]
            
            match = re.search(r'assert (\w+)\(', test_code)
            expected_name = match.group(1) if match else ""

            response, stopped = generate_fixed(model, tok, instr, device, max_new_tokens=128)
            clean_resp = response.replace(tok.eos_token, "").strip()

            valid = check_syntax(clean_resp)
            rep   = check_repetition(response)
            
            func_match = check_function_name(clean_resp, expected_name)
            test_pass = check_execution(clean_resp, test_code) if valid else False

            if valid:   syntax_count  += 1
            if rep:     rep_count     += 1
            if stopped: eos_stop_count += 1
            if func_match: func_name_match_count += 1
            if test_pass: test_pass_count += 1

            res = {
                "instruction": instr,
                "expected_name": expected_name,
                "generated":   response,
                "syntax_valid": valid,
                "repetitive":   rep,
                "stopped_at_eos": stopped,
                "func_name_match": func_match,
                "test_pass": test_pass
            }
            if i < 3:
                samples.append(res)

        print(f"=== STAGE 2B EVALUATION ({set_name}) ===")
        print(f"Checkpoint          : {ckpt_path}")
        print(f"Syntax Valid        : {syntax_count}/{n}  ({syntax_count/n*100:.1f}%)")
        print(f"Stopped at EOS      : {eos_stop_count}/{n}  ({eos_stop_count/n*100:.1f}%)")
        print(f"Repetition Rate     : {rep_count}/{n}  ({rep_count/n*100:.1f}%)")
        print(f"Func Name Match     : {func_name_match_count}/{n}  ({func_name_match_count/n*100:.1f}%)")
        print(f"Test Pass Rate      : {test_pass_count}/{n}  ({test_pass_count/n*100:.1f}%)")
        print("--- 3 Sample Generations ---")
        for j, s in enumerate(samples, 1):
            print(f"\n[{j}] {s['instruction']}")
            print(f"    syntax={s['syntax_valid']}  func_match={s['func_name_match']}  "
                  f"test_pass={s['test_pass']}  eos_stop={s['stopped_at_eos']}")
            print(f"    Output:\n{s['generated']}")

    evaluate_set(UNSEEN_PROMPTS, "EXPLICIT")
    evaluate_set(UNSEEN_PROMPTS_NATURAL, "NATURAL")

if __name__ == "__main__":
    main()
