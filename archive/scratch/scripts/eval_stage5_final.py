import os, sys, json, random, re
import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
import traceback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")

EVALS = [
    {"name": "Stage 3 Paraphrase", "file": os.path.join(ROOT, "data", "stage3_paraphrase_eval.jsonl")},
    {"name": "Stage 2E Adversarial", "file": os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")},
    {"name": "Stage 4B Name Copy", "file": os.path.join(ROOT, "data", "stage4b_name_copy_eval.jsonl")}
]

# We also need to build:
# C. Stage 5 teacher-style holdout
# D. HumanEval first 5

def generate_stage5_holdout():
    import random
    from prepare_stage5 import TASKS
    prompts = []
    random.seed(99999) # different from training data
    for i in range(500):
        task = random.choice(TASKS)
        has_name = random.random() < 0.60
        if has_name:
            fn = random.choice([task["name"]] + task["alt_names"])
            p = random.choice(task["prompts"]).replace("{func_name}", fn)
        else:
            fn = task.get("enforce_name", None)
            p = random.choice(task["no_name_prompts"])
            
        prompts.append({
            "id": f"stage5_eval_{i}",
            "prompt": p,
            "function_name": fn,
            "task_type": task["type"],
            "tests": task["tests"]
        })
    return prompts

def load_humaneval_5():
    # If the hf dataset fails to load due to no internet, hardcode the first 5
    return [
        {
            "id": "he_0",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0], 0.3)\n    True\n    \"\"\"\n",
            "tests": ["assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True", "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False"],
            "function_name": "has_close_elements"
        },
        {
            "id": "he_1",
            "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "tests": ["assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']"],
            "function_name": "separate_paren_groups"
        },
        {
            "id": "he_2",
            "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "tests": ["assert abs(truncate_number(3.5) - 0.5) < 1e-6"],
            "function_name": "truncate_number"
        },
        {
            "id": "he_3",
            "prompt": "from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
            "tests": ["assert below_zero([1, 2, -4, 5]) == True", "assert below_zero([1, 2, 3]) == False"],
            "function_name": "below_zero"
        },
        {
            "id": "he_4",
            "prompt": "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
            "tests": ["assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6"],
            "function_name": "mean_absolute_deviation"
        }
    ]

def run_eval(name, data, model, tok, prefix=""):
    print(f"\nEvaluating: {name} ({len(data)} examples)")
    
    results = []
    
    for i, ex in enumerate(data):
        prompt_str = f"<|im_start|>user\n{ex['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
        if "he_" in str(ex.get("id", "")):
             prompt_str = ex["prompt"]
             
        inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        
        stopped = False
        with torch.no_grad():
            for _ in range(192):
                sl, _ = model(inp_ids)
                nxt = torch.argmax(sl[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                if nxt == tok.eos_token_id:
                    stopped = True
                    break
                    
        raw = tok.decode(inp_ids[0], skip_special_tokens=False)
        if "he_" in str(ex.get("id", "")):
             gen = raw[len(ex["prompt"]):].replace(tok.eos_token, "").strip()
        else:
             gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").strip()
        
        syntax_ok = False
        try:
            ast.parse(gen)
            syntax_ok = True
        except:
            pass
            
        gen_fn = "NONE"
        import re
        if m := re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", gen):
            gen_fn = m.group(1)
            
        exp_fn = ex.get("function_name")
        name_match = (gen_fn == exp_fn) if exp_fn else None
        
        test_pass = False
        err_msg = ""
        ns = {}
        if syntax_ok and "tests" in ex:
            try:
                exec(compile(gen, "<string>", "exec"), ns)
                fn_to_test = exp_fn if exp_fn else gen_fn
                
                tests_passed = True
                for t in ex["tests"]:
                    t_eval = t.replace("{func_name}", fn_to_test)
                    try:
                        exec(compile(t_eval, "<string>", "exec"), ns)
                    except Exception as e:
                        tests_passed = False
                        err_msg = f"Test fail: {type(e).__name__}"
                        break
                test_pass = tests_passed
            except NameError as e:
                err_msg = f"NameError: {e}"
            except Exception as e:
                err_msg = f"Exec error: {type(e).__name__}"
                
        results.append({
            "id": ex.get("id", i),
            "prompt": ex["prompt"],
            "target": ex.get("target", ex.get("teacher_target", "")),
            "gen": gen,
            "syntax_ok": syntax_ok,
            "stopped": stopped,
            "test_pass": test_pass,
            "name_match": name_match,
            "err_msg": err_msg,
            "exp_fn": exp_fn,
            "gen_fn": gen_fn
        })
        
        if (i+1) % 50 == 0:
            print(f"  Done {i+1}/{len(data)}")
            
    syn_rate = sum(1 for r in results if r["syntax_ok"]) / len(results)
    eos_rate = sum(1 for r in results if r["stopped"]) / len(results)
    test_rate = sum(1 for r in results if r["test_pass"]) / len(results)
    
    nm_total = sum(1 for r in results if r["name_match"] is not None)
    nm_rate = sum(1 for r in results if r["name_match"]) / nm_total if nm_total > 0 else 0.0
    
    ne_rate = sum(1 for r in results if "NameError" in r["err_msg"]) / len(results)
    rep_rate = sum(1 for r in results if len(set(r["gen"].split())) < len(r["gen"].split()) * 0.3) / len(results)
    
    print(f"\n--- {name} Results ---")
    print(f"Syntax valid rate: {syn_rate:.1%}")
    print(f"EOS stop rate: {eos_rate:.1%}")
    print(f"Unit test pass rate: {test_rate:.1%}")
    if nm_total > 0:
        print(f"Function-name match: {nm_rate:.1%}")
    print(f"NameError rate: {ne_rate:.1%}")
    print(f"Repetition rate: {rep_rate:.1%}")
    
    with open(f"results/eval_{prefix}{name.replace(' ', '_').lower()}.json", "w") as f:
        json.dump(results, f, indent=2)

def main():
    print(f"Loading tokenizer and model...")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="samatnext", choices=["samatnext", "transformer"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")
    # For backwards compatibility with old usage `--baseline`
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    
    use_baseline = args.baseline or args.model == "transformer"
    
    if use_baseline:
        print("Evaluating BASELINE TRANSFORMER")
        from baseline.config import TransformerConfig
        from baseline.model import TransformerForCausalLM
        config = TransformerConfig(vocab_size=len(tok))
        model = TransformerForCausalLM(config).to(DEVICE)
        ckpt_path = args.ckpt if args.ckpt else os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_stage5_best.pt")
    else:
        print("Evaluating SAMAT_NEXT")
        config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
        model  = SamatNextForCausalLM(config).to(DEVICE)
        ckpt_path = args.ckpt if args.ckpt else CKPT
    
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    import ast # ensure ast is available in run_eval
    global ast
    import ast
    
    prefix = args.prefix
    if use_baseline and not prefix:
        prefix = "baseline_"
    
    # 1. Existing datasets (A, B, E)
    for ev in EVALS:
        if os.path.exists(ev["file"]):
            data = [json.loads(l) for l in open(ev["file"])][:500] # Cap at 500 for speed
            run_eval(ev["name"], data, model, tok, prefix=prefix)
            
    # 2. Stage 5 Teacher-style holdout (C)
    stage5_holdout = generate_stage5_holdout()
    if stage5_holdout:
        run_eval("Stage 5 Teacher-Style Holdout", stage5_holdout, model, tok, prefix=prefix)
            
    # 3. HumanEval 5 (D)
    he5 = load_humaneval_5()
    if he5:
        run_eval("HumanEval 5", he5, model, tok, prefix=prefix)

if __name__ == "__main__":
    main()
