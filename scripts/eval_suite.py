import os, sys, json, ast, re, subprocess
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EVALS = [
    {"name": "Stage 3 Paraphrase", "file": os.path.join(ROOT, "data", "stage3_paraphrase_eval.jsonl")},
    {"name": "Stage 2E Adversarial", "file": os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")},
    {"name": "Stage 4B Name Copy", "file": os.path.join(ROOT, "data", "stage4b_name_copy_eval.jsonl")}
]

def generate_stage5_holdout():
    import random
    from scripts.prepare_stage5 import TASKS
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

def run_test_worker_subprocess(gen, fn_to_test, tests_to_run, timeout_val):
    code_to_run = """
import sys, json
try:
    input_data = json.loads(sys.stdin.read())
    gen = input_data["gen"]
    tests = input_data["tests"]
    fn = input_data["fn"]
    ns = {}
    exec(compile(gen, "<string>", "exec"), ns)
    for t in tests:
        t_eval = t.replace("{func_name}", fn)
        exec(compile(t_eval, "<string>", "exec"), ns)
    print("SUCCESS")
except AssertionError:
    print("AssertionError")
except NameError:
    print("NameError")
except Exception as e:
    print(type(e).__name__)
"""
    input_dict = {"gen": gen, "tests": tests_to_run, "fn": fn_to_test}
    try:
        res = subprocess.run(
            [sys.executable, "-c", code_to_run],
            input=json.dumps(input_dict),
            capture_output=True, text=True, timeout=timeout_val
        )
        out = res.stdout.strip().split('\\n')[-1] if res.stdout.strip() else ""
        if "SUCCESS" in out:
            return True, ""
        elif "AssertionError" in out:
            return False, "AssertionError"
        elif "NameError" in out:
            return False, "NameError"
        elif out:
            return False, out
        else:
            if "SyntaxError" in res.stderr:
                 return False, "SyntaxError"
            return False, "Unknown Error"
    except subprocess.TimeoutExpired:
        return False, "Timeout"

def evaluate_subset(name, data, model, tok, device, return_details=False, timeout_seconds=None, dtype=torch.float32):
    results = []
    
    for i, ex in enumerate(data):
        if "Stage 6C" in name:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}\n<|im_end|>\n<|im_start|>assistant\n"
        elif "he_" in str(ex.get("id", "")):
            # HumanEval Chat-formatted FULL wrapper (user specified in instructions)
            prompt_str = f"<|im_start|>user\nWrite the complete Python function for the following task.\n\n{ex['prompt']}\n\nReturn only Python code. No markdown. No explanation.\n<|im_end|>\n<|im_start|>assistant\n"
        elif "Stage 3" in name or "Stage 5" in name or "Stage 2E" in name or "Stage 4B" in name:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
        else:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
             
        inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        stopped_im_end = False
        stopped_eot = False
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                for _ in range(192):
                    sl, _ = model(inp_ids)
                    nxt = torch.argmax(sl[0, -1, :]).item()
                    inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                    if nxt == 151645:  # <|im_end|>
                        stopped_im_end = True
                        break
                    elif nxt == tok.eos_token_id:  # <|endoftext|>
                        stopped_eot = True
                        break
                    
        raw = tok.decode(inp_ids[0], skip_special_tokens=False)
        gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").replace("<|im_end|>", "").strip()
        
        # handle markdown formatting if the model slipped some in (common in Stage 6 generation)
        if gen.startswith("```"):
            gen = gen.split("\n", 1)[-1]
            if gen.endswith("```"):
                gen = gen[:-3]
        if gen.startswith("python\n"):
            gen = gen[7:]
        gen = gen.strip()
        
        eval_gen = gen
        if "Completion" in name or "he_" in str(ex.get("id", "")):
            eval_gen = ex["prompt"] + "\n" + gen

        syntax_ok = False
        try:
            ast.parse(eval_gen)
            syntax_ok = True
        except:
            pass
            
        gen_fn = "NONE"
        if m := re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", gen):
            gen_fn = m.group(1)
            
        exp_fn = ex.get("function_name")
        
        test_pass = False
        err_msg = ""
        if syntax_ok and ("tests" in ex or "hidden_tests" in ex):
            fn_to_test = exp_fn if exp_fn else gen_fn
            tests_to_run = ex.get("tests", []) + ex.get("hidden_tests", [])
            if timeout_seconds is not None:
                timeout_val = timeout_seconds
            else:
                timeout_val = 2.0 if ("HumanEval" in name or "he_" in name) else 1.0
            test_pass, err_msg = run_test_worker_subprocess(eval_gen, fn_to_test, tests_to_run, timeout_val)
                
        results.append({
            "syntax_ok": syntax_ok,
            "stopped_im_end": stopped_im_end,
            "stopped_eot": stopped_eot,
            "test_pass": test_pass,
            "err_msg": err_msg,
            "gen": gen,
            "prompt": prompt_str,
            "raw": raw,
            "eval_gen": eval_gen,
            "tests": ex.get("tests", []) + ex.get("hidden_tests", [])
        })
            
    syn_rate = sum(1 for r in results if r["syntax_ok"]) / len(results)
    im_end_rate = sum(1 for r in results if r["stopped_im_end"]) / len(results)
    eot_rate = sum(1 for r in results if r["stopped_eot"]) / len(results)
    eos_rate = sum(1 for r in results if r["stopped_im_end"] or r["stopped_eot"]) / len(results)
    test_rate = sum(1 for r in results if r["test_pass"]) / len(results)
    rep_rate = sum(1 for r in results if len(set(r["gen"].split())) < len(r["gen"].split()) * 0.3) / len(results) if results else 0
    
    timeout_rate = sum(1 for r in results if r["err_msg"] == "Timeout") / len(results)
    assert_rate = sum(1 for r in results if r["err_msg"] == "AssertionError") / len(results)
    name_rate = sum(1 for r in results if r["err_msg"] == "NameError") / len(results)
    exc_rate = sum(1 for r in results if r["err_msg"] not in ["", "Timeout", "AssertionError", "NameError"]) / len(results)
    
    summary = {
        "pass_rate": test_rate,
        "syntax_rate": syn_rate,
        "eos_rate": eos_rate,
        "emitted_im_end_rate": im_end_rate,
        "emitted_endoftext_rate": eot_rate,
        "rep_rate": rep_rate,
        "timeout_rate": timeout_rate,
        "assert_rate": assert_rate,
        "name_rate": name_rate,
        "exc_rate": exc_rate
    }
    
    if return_details:
        return summary, results
    return summary

def run_all_evals(model, tok, device, dtype=torch.float32):
    metrics = {}
    
    # 1. Existing datasets
    for ev in EVALS:
        if os.path.exists(ev["file"]):
            data = [json.loads(l) for l in open(ev["file"])]
            # Stage 2E and Stage 4B, Stage 3 can be capped at 200 for speed
            if len(data) > 200:
                data = data[:200]
            if ev["name"] == "Stage 6A Holdout":
                # Ensure we evaluate on the actual holdout length
                pass
            res = evaluate_subset(ev["name"], data, model, tok, device, dtype=dtype)
            metrics[ev["name"]] = res
            
    # 2. Stage 5
    stage5_data = generate_stage5_holdout()[:200]
    res = evaluate_subset("Stage 5", stage5_data, model, tok, device, dtype=dtype)
    metrics["Stage 5"] = res
    
    # 3. HumanEval 5
    he5 = load_humaneval_5()
    res = evaluate_subset("HumanEval 5", he5, model, tok, device, dtype=dtype)
    metrics["HumanEval 5"] = res
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"[{k}] Pass: {v['pass_rate']:.1%} | Syn: {v['syntax_rate']:.1%} | EOS: {v['eos_rate']:.1%} (im_end={v['emitted_im_end_rate']:.1%} eot={v['emitted_endoftext_rate']:.1%}) | Rep: {v['rep_rate']:.1%} | TO: {v['timeout_rate']:.1%} | Assrt: {v['assert_rate']:.1%} | NErr: {v['name_rate']:.1%} | Exc: {v['exc_rate']:.1%}")
    print("="*50 + "\n")
    return metrics
