import os, sys, json, ast, re, torch
from collections import defaultdict
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

DATA_DIR = os.path.join(ROOT, "data")
CKPTS = {
    "Stage 3": os.path.join(ROOT, "checkpoints", "samat_next_350m_stage3_best.pt"),
    "Stage 4": os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4_best.pt"),
    "Stage 4B": os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4b_best.pt"),
}
SETS = [
    {"name": "Stage 3 Holdout", "file": os.path.join(DATA_DIR, "stage3_paraphrase_eval.jsonl")},
    {"name": "Stage 2E Adv", "file": os.path.join(DATA_DIR, "stage2e_adversarial_holdout.jsonl")},
]

def check_syntax(code):
    try: ast.parse(code); return True
    except: return False

def check_exec(code, tests, exp_fn):
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
    except Exception:
        return False
    if exp_fn and exp_fn not in ns:
        return False
    for t in tests:
        try: exec(compile(t, "<string>", "exec"), ns)
        except: return False
    return True

def extract_fn(code):
    m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code)
    return m.group(1) if m else None

def evaluate(model, tok, device, data_sets):
    res = {}
    model.eval()
    eos_id = tok.eos_token_id
    marker = "<|im_start|>assistant\n"
    
    for s in data_sets:
        if not os.path.exists(s["file"]): continue
        all_data = [json.loads(l) for l in open(s["file"], encoding="utf-8")]
        
        # Quick eval: 100 samples per dataset to save time
        import random
        random.seed(42)
        data = random.sample(all_data, min(100, len(all_data)))
        
        syn_ok = eos_ok = pass_ok = total = 0
        for ex in data:
            total += 1
            prompt = ex["prompt"]
            exp_fn = ex.get("function_name")
            tests = ex.get("tests", [])
            
            inp = tok(f"<|im_start|>user\n{prompt}{tok.eos_token}\n{marker}", 
                      add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            stopped = False
            with torch.no_grad():
                for _ in range(128):
                    logits, _ = model(inp)
                    nxt = torch.argmax(logits[0,-1,:]).item()
                    inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], 1)
                    if nxt == eos_id: stopped = True; break
            
            raw = tok.decode(inp[0], skip_special_tokens=False)
            gen = raw.split(marker)[-1].replace(tok.eos_token, "").strip()
            
            if stopped: eos_ok += 1
            if check_syntax(gen): syn_ok += 1
            if check_exec(gen, tests, exp_fn): pass_ok += 1
            
        res[s["name"]] = {"total": total, "syntax": syn_ok, "eos": eos_ok, "pass": pass_ok}
    return res

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint Selection Eval | device={device}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model = SamatNextForCausalLM(config).to(device)

    all_res = {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"[{name}] MISSING: {path}")
            continue
        print(f"\n--- Loading {name} ---")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        all_res[name] = evaluate(model, tok, device, SETS)
        
        for sname, v in all_res[name].items():
            t = v['total']
            print(f"{sname}: Syntax {v['syntax']/t*100:.1f}% | EOS {v['eos']/t*100:.1f}% | Pass {v['pass']/t*100:.1f}%")

if __name__ == "__main__":
    main()
