"""
Stage 3 Evaluation Script
Evaluates on:
  A. stage3_paraphrase_eval.jsonl (paraphrase holdout)
  B. stage2e_adversarial_holdout.jsonl (Stage 2E re-test)

Metrics:
  - syntax valid rate
  - EOS stop rate
  - function-name match rate
  - unit test pass rate
  - wrong-task rate
  - NameError rate
  - pass rate by paraphrase_style
  - pass rate by task category
  - 30 failed examples (prompt, generated code, failing test, error)
  - 30 successful examples (prompt, generated code, passed tests)
"""
import os
import sys
import json
import re
import ast
import argparse
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_stage2a_fixed import generate_fixed


def check_syntax(code_str):
    try:
        ast.parse(code_str)
        return True, "OK"
    except SyntaxError as e:
        return False, "SyntaxError"
    except Exception as e:
        return False, type(e).__name__


def check_execution(code_str, tests):
    try:
        ns = {}
        exec(compile(code_str, "<string>", "exec"), ns)
        for t in tests:
            exec(compile(t, "<string>", "exec"), ns)
        return True, "Pass", None
    except Exception as e:
        return False, type(e).__name__, str(e)


def get_function_name(code_str):
    try:
        m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code_str)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def evaluate_set(model, tok, device, items, label):
    n = len(items)
    syntax_count = eos_count = fn_match_count = test_pass_count = 0
    fail_type_counts = Counter()
    category_stats   = defaultdict(lambda: {"pass": 0, "total": 0})
    style_stats      = defaultdict(lambda: {"pass": 0, "total": 0})
    failures, successes = [], []

    print(f"\nEvaluating {n} prompts ({label})...")
    for i, item in enumerate(items):
        prompt     = item["prompt"]
        exp_fn     = item["function_name"]
        tests      = item["tests"]
        category   = item.get("category", "unknown")
        style      = item.get("paraphrase_style", "unknown")[:40]

        response, stopped = generate_fixed(model, tok, prompt, device, max_new_tokens=192)
        generated = response.replace(tok.eos_token, "").strip()

        valid, syn_err = check_syntax(generated)
        actual_fn = get_function_name(generated)
        fn_match = (actual_fn == exp_fn)
        test_pass, exec_err, exec_msg = False, "SyntaxError", None

        if valid:
            test_pass, exec_err, exec_msg = check_execution(generated, tests)

        if valid:        syntax_count  += 1
        if stopped:      eos_count     += 1
        if fn_match:     fn_match_count += 1
        if test_pass:
            test_pass_count += 1
        else:
            err = exec_err if valid else syn_err
            fail_type_counts[err] += 1

        category_stats[category]["total"] += 1
        style_stats[style]["total"] += 1
        if test_pass:
            category_stats[category]["pass"] += 1
            style_stats[style]["pass"] += 1

        record = {
            "prompt": prompt,
            "expected_fn": exp_fn,
            "generated": generated,
            "syntax_valid": valid,
            "fn_match": fn_match,
            "test_pass": test_pass,
            "eos_stop": stopped,
            "error": exec_err if not test_pass else "None",
            "exec_msg": exec_msg or "",
            "tests": tests,
        }
        if not test_pass and len(failures) < 30:
            failures.append(record)
        if test_pass and len(successes) < 30:
            successes.append(record)

    name_err_count  = fail_type_counts.get("NameError", 0)
    wrong_task_count = sum(cnt for err, cnt in fail_type_counts.items()
                           if err not in ["NameError", "SyntaxError", "None"])

    print(f"\n{'='*60}")
    print(f"EVALUATION: {label}")
    print(f"{'='*60}")
    print(f"Syntax Valid        : {syntax_count}/{n}  ({syntax_count/n*100:.1f}%)")
    print(f"Stopped at EOS      : {eos_count}/{n}  ({eos_count/n*100:.1f}%)")
    print(f"Func Name Match     : {fn_match_count}/{n}  ({fn_match_count/n*100:.1f}%)")
    print(f"Unit Test Pass Rate : {test_pass_count}/{n}  ({test_pass_count/n*100:.1f}%)")
    print(f"NameError Rate      : {name_err_count}/{n}  ({name_err_count/n*100:.1f}%)")
    print(f"Wrong-Task Rate     : {wrong_task_count}/{n}  ({wrong_task_count/n*100:.1f}%)")

    print("\nFailures by error type:")
    for err, cnt in fail_type_counts.most_common():
        print(f"  {err}: {cnt}")

    print("\nPass rate by task category:")
    for cat, stats in sorted(category_stats.items()):
        p, t = stats["pass"], stats["total"]
        print(f"  {cat:<30} {p}/{t}  ({p/t*100:.1f}%)")

    print("\nPass rate by paraphrase style:")
    style_rows = sorted(style_stats.items(), key=lambda x: -x[1]["pass"] / max(x[1]["total"], 1))
    for style, stats in style_rows:
        p, t = stats["pass"], stats["total"]
        pct = p / t * 100 if t else 0
        print(f"  [{pct:5.1f}%]  {style}")

    print(f"\n--- 30 FAILED EXAMPLES ---")
    for j, r in enumerate(failures, 1):
        print(f"\n[F{j}] {r['prompt']}")
        print(f"     Expected fn: {r['expected_fn']} | Error: {r['error']}")
        if r["exec_msg"]:
            print(f"     Msg: {r['exec_msg'][:80]}")
        print(f"     Generated:\n{r['generated']}")

    print(f"\n--- 30 SUCCESSFUL EXAMPLES ---")
    for j, r in enumerate(successes, 1):
        print(f"\n[S{j}] {r['prompt']}")
        print(f"     Generated:\n{r['generated']}")

    return {
        "syntax": syntax_count / n,
        "eos_stop": eos_count / n,
        "fn_match": fn_match_count / n,
        "test_pass": test_pass_count / n,
        "name_error": name_err_count / n,
        "wrong_task": wrong_task_count / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    ckpt_path = args.ckpt or os.path.join(ROOT, "checkpoints", "samat_next_350m_stage3_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # --- Set A: Stage 3 paraphrase holdout ---
    stage3_eval_path = os.path.join(ROOT, "data", "stage3_paraphrase_eval.jsonl")
    stage3_items = []
    with open(stage3_eval_path, encoding="utf-8") as f:
        for line in f:
            stage3_items.append(json.loads(line))

    results_a = evaluate_set(model, tok, device, stage3_items, "Stage 3 Paraphrase Holdout")

    # --- Set B: Stage 2E adversarial re-test ---
    stage2e_path = os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")
    stage2e_items = []
    with open(stage2e_path, encoding="utf-8") as f:
        for line in f:
            stage2e_items.append(json.loads(line))

    results_b = evaluate_set(model, tok, device, stage2e_items, "Stage 2E Adversarial Re-test")

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY vs SUCCESS TARGETS")
    print("="*60)
    targets = {
        "Stage 3 holdout unit test pass ≥ 70%":   results_a["test_pass"] >= 0.70,
        "Stage 2E retest unit test pass ≥ 40%":    results_b["test_pass"] >= 0.40,
        "Function-name match ≥ 80% (Stage 3)":    results_a["fn_match"]  >= 0.80,
        "EOS stop rate ≥ 95% (Stage 3)":           results_a["eos_stop"]  >= 0.95,
    }
    for desc, met in targets.items():
        status = "PASS ✓" if met else "FAIL ✗"
        print(f"  {status}  {desc}")


if __name__ == "__main__":
    main()
