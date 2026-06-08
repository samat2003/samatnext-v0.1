"""
Stage 4 Evaluation: Arbitrary Function-Name Copy Binding
========================================================
Runs 3 evaluation sets:
  A. Stage 4 name-binding holdout (1,000 examples)
  B. Stage 3 paraphrase holdout (500 examples)
  C. Stage 2E adversarial retest (300 examples)

Reports:
- Syntax Valid rate
- EOS stop rate
- Function-name match rate
- Unit test pass rate
- NameError rate
- Wrong-task rate
- Pass rate by name style (for Stage 4)
- Pass rate by task category
"""
import os, sys, json, ast, torch, re
from collections import defaultdict
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

DATA_DIR = os.path.join(ROOT, "data")
CKPT_FILE = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4_best.pt")

SETS = [
    {
        "name": "Stage 4 Name-Binding Holdout",
        "file": os.path.join(DATA_DIR, "stage4_name_binding_eval.jsonl"),
        "targets": {"fn_match": 0.90, "test_pass": 0.75, "name_error_max": 0.10}
    },
    {
        "name": "Stage 3 Paraphrase Holdout",
        "file": os.path.join(DATA_DIR, "stage3_paraphrase_eval.jsonl"),
        "targets": {"test_pass": 0.70}
    },
    {
        "name": "Stage 2E Adversarial Re-test",
        "file": os.path.join(DATA_DIR, "stage2e_adversarial_holdout.jsonl"),
        "targets": {"fn_match": 0.50, "test_pass": 0.40}
    }
]

def check_syntax(code):
    try: ast.parse(code); return True
    except SyntaxError: return False

def check_execution(code, tests, expected_fn):
    # Fails if syntax error, name error, or unit test fail
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
    except SyntaxError:
        return False, "SyntaxError", "SyntaxError in generation"
    except Exception as e:
        return False, type(e).__name__, str(e)
    
    if expected_fn not in ns:
        return False, "NameError", f"name '{expected_fn}' is not defined"
    
    for t in tests:
        try:
            exec(compile(t, "<string>", "exec"), ns)
        except NameError as e:
            return False, "NameError", str(e)
        except Exception as e:
            return False, type(e).__name__, str(e)
    return True, "Success", ""

def extract_fn_name(code):
    m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code)
    return m.group(1) if m else None

def evaluate_set(model, tok, device, set_info):
    file_path = set_info["file"]
    if not os.path.exists(file_path):
        print(f"\nMissing dataset: {file_path}")
        return None
    
    data = [json.loads(x) for x in open(file_path, encoding="utf-8")]
    print(f"\nEvaluating {len(data)} prompts ({set_info['name']})...")

    eos_id, eos_tok = tok.eos_token_id, tok.eos_token
    marker = "<|im_start|>assistant\n"
    
    metrics = {
        "total": len(data),
        "syntax": 0,
        "eos": 0,
        "fn_match": 0,
        "test_pass": 0,
        "wrong_task": 0,
        "errors": defaultdict(int),
        "cat_pass": defaultdict(int),
        "cat_total": defaultdict(int),
        "style_pass": defaultdict(int),
        "style_total": defaultdict(int),
        "len_pass": defaultdict(int),
        "len_total": defaultdict(int),
        "failed_examples": [],
        "success_examples": []
    }
    
    model.eval()
    for idx, ex in enumerate(data):
        prompt = ex["prompt"]
        exp_fn = ex["function_name"]
        cat    = ex.get("category", "unknown")
        style  = ex.get("name_style", "unknown")
        nlen   = str(len(exp_fn.split("_"))) + "-token"
        tests  = ex["tests"]

        inp_ids = tok(f"<|im_start|>user\n{prompt}{eos_tok}\n{marker}", 
                      add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        stopped = False
        with torch.no_grad():
            for _ in range(160):
                logits, _ = model(inp_ids)
                nxt = torch.argmax(logits[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == eos_id:
                    stopped = True; break
                    
        raw = tok.decode(inp_ids[0], skip_special_tokens=False)
        gen = raw.split(marker)[-1].replace(eos_tok, "").strip()

        if stopped: metrics["eos"] += 1
        if check_syntax(gen): metrics["syntax"] += 1
        
        gen_fn = extract_fn_name(gen)
        if gen_fn == exp_fn: metrics["fn_match"] += 1
        
        passed, err_type, err_msg = check_execution(gen, tests, exp_fn)
        
        metrics["cat_total"][cat] += 1
        metrics["style_total"][style] += 1
        metrics["len_total"][nlen] += 1
        
        if passed:
            metrics["test_pass"] += 1
            metrics["cat_pass"][cat] += 1
            metrics["style_pass"][style] += 1
            metrics["len_pass"][nlen] += 1
            if len(metrics["success_examples"]) < 20:
                metrics["success_examples"].append((prompt, exp_fn, gen))
        else:
            metrics["errors"][err_type] += 1
            if err_type not in ["NameError", "SyntaxError", "TypeError"]:
                metrics["wrong_task"] += 1
            if len(metrics["failed_examples"]) < 20:
                metrics["failed_examples"].append((prompt, exp_fn, gen, err_type, err_msg))
                
    return metrics

def print_metrics(set_info, metrics):
    if not metrics: return
    T = metrics["total"]
    print(f"\n============================================================")
    print(f"EVALUATION: {set_info['name']}")
    print(f"============================================================")
    print(f"Syntax Valid        : {metrics['syntax']}/{T}  ({metrics['syntax']/T*100:.1f}%)")
    print(f"Stopped at EOS      : {metrics['eos']}/{T}  ({metrics['eos']/T*100:.1f}%)")
    print(f"Func Name Match     : {metrics['fn_match']}/{T}  ({metrics['fn_match']/T*100:.1f}%)")
    print(f"Unit Test Pass Rate : {metrics['test_pass']}/{T}  ({metrics['test_pass']/T*100:.1f}%)")
    print(f"NameError Rate      : {metrics['errors']['NameError']}/{T}  ({metrics['errors']['NameError']/T*100:.1f}%)")
    print(f"Wrong-Task Rate     : {metrics['wrong_task']}/{T}  ({metrics['wrong_task']/T*100:.1f}%)")
    
    print(f"\nFailures by error type:")
    for k, v in sorted(metrics["errors"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")

    print(f"\nPass rate by task category:")
    for c in sorted(metrics["cat_total"].keys()):
        p = metrics["cat_pass"][c]; t = metrics["cat_total"][c]
        print(f"  {c:<30} {p}/{t}  ({p/t*100:.1f}%)")

    print(f"\nPass rate by name length:")
    for c in sorted(metrics["len_total"].keys()):
        p = metrics["len_pass"][c]; t = metrics["len_total"][c]
        print(f"  {c:<12} {p}/{t}  ({p/t*100:.1f}%)")

    print(f"\nPass rate by name style:")
    for c in sorted(metrics["style_total"].keys()):
        p = metrics["style_pass"][c]; t = metrics["style_total"][c]
        print(f"  [{p/t*100:>5.1f}%]  {c}")

    print(f"\n--- {len(metrics['failed_examples'])} FAILED EXAMPLES ---")
    for i, (prompt, exp_fn, gen, err_type, err_msg) in enumerate(metrics["failed_examples"], 1):
        print(f"\n[F{i}] {prompt}")
        print(f"     Expected fn: {exp_fn} | Error: {err_type}")
        print(f"     Msg: {err_msg}")
        print(f"     Generated:\n{gen}")

    print(f"\n--- {len(metrics['success_examples'])} SUCCESSFUL EXAMPLES ---")
    for i, (prompt, exp_fn, gen) in enumerate(metrics["success_examples"], 1):
        print(f"\n[S{i}] {prompt}")
        print(f"     Generated:\n{gen}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage 4 Model on device: {device}")
    
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        
    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model = SamatNextForCausalLM(config).to(device)
    
    if not os.path.exists(CKPT_FILE):
        print(f"ERROR: Checkpoint not found: {CKPT_FILE}")
        sys.exit(1)
        
    print(f"Checkpoint: {CKPT_FILE}")
    model.load_state_dict(torch.load(CKPT_FILE, map_location=device, weights_only=True))

    all_results = {}
    for s in SETS:
        m = evaluate_set(model, tok, device, s)
        if m:
            print_metrics(s, m)
            all_results[s["name"]] = m
            
    print("\n============================================================")
    print("SUMMARY vs SUCCESS TARGETS")
    print("============================================================")
    
    # Check targets
    for s in SETS:
        if s["name"] not in all_results: continue
        m = all_results[s["name"]]
        T = m["total"]
        targets = s["targets"]
        print(f"\n{s['name']}:")
        if "fn_match" in targets:
            val = m["fn_match"] / T
            mark = "PASS" if val >= targets["fn_match"] else "FAIL"
            print(f"  [{mark}] Func-name match: {val*100:.1f}% (target >= {targets['fn_match']*100:.0f}%)")
        if "test_pass" in targets:
            val = m["test_pass"] / T
            mark = "PASS" if val >= targets["test_pass"] else "FAIL"
            print(f"  [{mark}] Unit test pass : {val*100:.1f}% (target >= {targets['test_pass']*100:.0f}%)")
        if "name_error_max" in targets:
            val = m["errors"]["NameError"] / T
            mark = "PASS" if val < targets["name_error_max"] else "FAIL"
            print(f"  [{mark}] NameError rate : {val*100:.1f}% (target < {targets['name_error_max']*100:.0f}%)")

if __name__ == "__main__":
    main()
