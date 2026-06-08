"""
Stage 4B Evaluation: Explicit Identifier Copy Curriculum
=========================================================
Runs 4 evaluation sets:
  A. Stage 4B name-copy holdout (2,000)
  B. Stage 4  name-binding holdout (1,000)
  C. Stage 3  paraphrase holdout (500)
  D. Stage 2E adversarial retest (300)

Reports:
  - syntax valid rate
  - exact function-name match rate
  - unit test pass rate
  - NameError rate
  - pass rate by name length (2-token / 3-token / 4-token)
  - partial-copy examples (first token matched but not all)
  - 20 failures, 20 successes
"""
import os, sys, json, ast, re, torch
from collections import defaultdict
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

DATA_DIR  = os.path.join(ROOT, "data")
CKPT_FILE = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4b_best.pt")

SETS = [
    {
        "name":    "Stage 4B Name-Copy Holdout",
        "file":    os.path.join(DATA_DIR, "stage4b_name_copy_eval.jsonl"),
        "targets": {"fn_match": 0.75, "test_pass": 0.60, "name_error_max": 0.30},
    },
    {
        "name":    "Stage 4 Name-Binding Holdout",
        "file":    os.path.join(DATA_DIR, "stage4_name_binding_eval.jsonl"),
        "targets": {"fn_match": 0.40, "test_pass": 0.35},
    },
    {
        "name":    "Stage 3 Paraphrase Holdout",
        "file":    os.path.join(DATA_DIR, "stage3_paraphrase_eval.jsonl"),
        "targets": {"test_pass": 0.60},
    },
    {
        "name":    "Stage 2E Adversarial Re-test",
        "file":    os.path.join(DATA_DIR, "stage2e_adversarial_holdout.jsonl"),
        "targets": {"fn_match": 0.10, "test_pass": 0.10},
    },
]

def check_syntax(code):
    try: ast.parse(code); return True
    except: return False

def check_exec(code, tests, exp_fn):
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
    except SyntaxError:
        return False, "SyntaxError", ""
    except Exception as e:
        return False, type(e).__name__, str(e)
    if exp_fn not in ns:
        return False, "NameError", f"'{exp_fn}' not defined"
    for t in tests:
        try:
            exec(compile(t, "<string>", "exec"), ns)
        except Exception as e:
            return False, type(e).__name__, str(e)
    return True, "OK", ""

def extract_fn(code):
    m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code)
    return m.group(1) if m else None

def partial_copy(expected, got):
    """True if at least the first token of expected appears in got."""
    if got is None: return False
    toks_exp = expected.split("_")
    toks_got = got.split("_")
    return toks_exp[0] == toks_got[0] and got != expected

def evaluate(model, tok, device, s):
    fpath = s["file"]
    if not os.path.exists(fpath):
        print(f"\n[SKIP] Missing: {fpath}"); return None
    data = [json.loads(l) for l in open(fpath, encoding="utf-8")]
    print(f"\nEvaluating {len(data)} examples ({s['name']})...")

    eos_id  = tok.eos_token_id
    marker  = "<|im_start|>assistant\n"
    metrics = {
        "total": len(data), "syntax": 0, "eos": 0,
        "fn_match": 0, "test_pass": 0, "partial_copy": 0,
        "errors": defaultdict(int),
        "len_match": defaultdict(int), "len_total": defaultdict(int),
        "cat_match": defaultdict(int), "cat_total": defaultdict(int),
        "failed": [], "succeeded": [],
    }

    model.eval()
    for ex in data:
        prompt  = ex["prompt"]
        exp_fn  = ex["function_name"]
        tests   = ex.get("tests", [])
        nlen    = f"{len(exp_fn.split('_'))}-token"
        cat     = ex.get("task_type", ex.get("category", "unknown"))

        inp = tok(f"<|im_start|>user\n{prompt}{tok.eos_token}\n{marker}",
                  add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        stopped = False
        with torch.no_grad():
            for _ in range(192):
                logits, _ = model(inp)
                nxt = torch.argmax(logits[0,-1,:]).item()
                inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], 1)
                if nxt == eos_id: stopped = True; break

        raw = tok.decode(inp[0], skip_special_tokens=False)
        gen = raw.split(marker)[-1].replace(tok.eos_token, "").strip()

        if stopped: metrics["eos"] += 1
        if check_syntax(gen): metrics["syntax"] += 1

        gen_fn = extract_fn(gen)
        name_ok = (gen_fn == exp_fn)
        if name_ok: metrics["fn_match"] += 1
        if partial_copy(exp_fn, gen_fn): metrics["partial_copy"] += 1

        metrics["len_total"][nlen] += 1
        metrics["cat_total"][cat]  += 1

        passed, err_type, err_msg = check_exec(gen, tests, exp_fn)
        if passed:
            metrics["test_pass"] += 1
            metrics["len_match"][nlen] += 1
            metrics["cat_match"][cat]  += 1
            if len(metrics["succeeded"]) < 20:
                metrics["succeeded"].append((prompt, exp_fn, gen_fn, gen))
        else:
            metrics["errors"][err_type] += 1
            if len(metrics["failed"]) < 20:
                metrics["failed"].append((prompt, exp_fn, gen_fn, gen, err_type, err_msg))

    return metrics

def print_metrics(s, m):
    if not m: return
    T = m["total"]
    print(f"\n{'='*62}")
    print(f"EVALUATION: {s['name']}")
    print(f"{'='*62}")
    print(f"Syntax Valid        : {m['syntax']}/{T}  ({m['syntax']/T*100:.1f}%)")
    print(f"Stopped at EOS      : {m['eos']}/{T}  ({m['eos']/T*100:.1f}%)")
    print(f"Func Name Match     : {m['fn_match']}/{T}  ({m['fn_match']/T*100:.1f}%)")
    print(f"Unit Test Pass      : {m['test_pass']}/{T}  ({m['test_pass']/T*100:.1f}%)")
    print(f"NameError Rate      : {m['errors']['NameError']}/{T}  ({m['errors']['NameError']/T*100:.1f}%)")
    print(f"Partial Copy        : {m['partial_copy']}/{T}  ({m['partial_copy']/T*100:.1f}%)")

    print(f"\nPass rate by name length:")
    for k in sorted(m["len_total"]):
        p = m["len_match"][k]; t = m["len_total"][k]
        print(f"  {k:<12} {p}/{t}  ({p/t*100:.1f}%)")

    print(f"\nPass rate by task category:")
    for k in sorted(m["cat_total"]):
        p = m["cat_match"][k]; t = m["cat_total"][k]
        print(f"  {k:<30} {p}/{t}  ({p/t*100:.1f}%)")

    print(f"\nErrors by type:")
    for k,v in sorted(m["errors"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\n--- {len(m['failed'])} FAILED EXAMPLES ---")
    for i, (prompt, exp, got, gen, err, msg) in enumerate(m["failed"], 1):
        print(f"\n[F{i}] {prompt[:90]}")
        print(f"     Expected: {exp} | Got: {got} | Error: {err}")
        print(f"     Code: {gen[:120]}")

    print(f"\n--- {len(m['succeeded'])} SUCCESSFUL EXAMPLES ---")
    for i, (prompt, exp, got, gen) in enumerate(m["succeeded"], 1):
        print(f"\n[S{i}] {prompt[:90]}")
        print(f"     Expected: {exp} | Got: {got}")
        print(f"     Code: {gen[:120]}")

def print_summary(all_results):
    print(f"\n{'='*62}")
    print(f"SUMMARY vs TARGETS")
    print(f"{'='*62}")
    for s in SETS:
        m = all_results.get(s["name"])
        if not m: continue
        T = m["total"]
        targets = s["targets"]
        print(f"\n{s['name']}:")
        if "fn_match" in targets:
            v = m["fn_match"]/T
            mark = "PASS" if v >= targets["fn_match"] else "FAIL"
            print(f"  [{mark}] Name match    : {v*100:.1f}%  (target >= {targets['fn_match']*100:.0f}%)")
        if "test_pass" in targets:
            v = m["test_pass"]/T
            mark = "PASS" if v >= targets["test_pass"] else "FAIL"
            print(f"  [{mark}] Test pass     : {v*100:.1f}%  (target >= {targets['test_pass']*100:.0f}%)")
        if "name_error_max" in targets:
            v = m["errors"]["NameError"]/T
            mark = "PASS" if v < targets["name_error_max"] else "FAIL"
            print(f"  [{mark}] NameError rate: {v*100:.1f}%  (target < {targets['name_error_max']*100:.0f}%)")
        # Name-length breakdown
        print(f"  Name-match by length:")
        for k in sorted(m["len_total"]):
            p = m["len_match"][k]; t = m["len_total"][k]
            print(f"    {k}: {p}/{t}  ({p/t*100:.1f}%)")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 4B Evaluation | device={device}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    if not os.path.exists(CKPT_FILE):
        print(f"ERROR: {CKPT_FILE} not found"); sys.exit(1)
    model.load_state_dict(torch.load(CKPT_FILE, map_location=device, weights_only=True))
    print(f"Loaded: {CKPT_FILE}")

    all_results = {}
    for s in SETS:
        m = evaluate(model, tok, device, s)
        if m:
            print_metrics(s, m)
            all_results[s["name"]] = m

    print_summary(all_results)

if __name__ == "__main__":
    main()
