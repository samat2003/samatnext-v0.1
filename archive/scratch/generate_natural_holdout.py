import sys, json, torch, os, ast, uuid, subprocess, tempfile, re
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 16

def extract_doctests(code):
    tests = []; lines = code.split('\n'); i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>>> '):
            expr = line[4:].strip()
            j = i + 1; expected = []
            while j < len(lines) and not lines[j].strip().startswith('>>>') and lines[j].strip() not in ('"""', "'''") and not lines[j].strip().startswith('def '):
                if lines[j].strip(): expected.append(lines[j].strip())
                j += 1
            if expected:
                exp_str = '\\n'.join(expected)
                tests.append(f"assert {expr} == {exp_str}")
            i = j
        else: i += 1
    return tests

def execute_with_timeout(code, tests):
    test_code = code + "\n" + "\n".join(tests)
    script_content = f"import math, collections, itertools, functools, re, string\n{test_code}\nprint('__SUCCESS__')\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        temp_path = f.name
    try:
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=2.0)
        if result.returncode == 0 and "__SUCCESS__" in result.stdout: return True, "Valid"
        else: return False, f"Test failed: {result.stderr.strip()[:100]}"
    except subprocess.TimeoutExpired: return False, "Timeout"
    except Exception as e: return False, str(e)[:100]
    finally: os.remove(temp_path)

def strip_fences(code):
    match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
    return match.group(1).strip() if match else code.strip()

def split_function(code):
    try: tree = ast.parse(code)
    except: return None, None
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef): return None, None
    func_def = tree.body[0]
    lines = code.split("\n")
    if ast.get_docstring(func_def) and isinstance(func_def.body[0], ast.Expr) and isinstance(func_def.body[0].value, ast.Constant):
        split_line = func_def.body[0].end_lineno
    else: split_line = func_def.body[0].lineno - 1
    return "\n".join(lines[:split_line]) + "\n", "\n".join(lines[split_line:])

def validate_task(code, tests, function_name):
    code = strip_fences(code)
    try: parsed = ast.parse(code)
    except: return False, "Syntax error"
    
    found_fname = None
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef):
            found_fname = node.name
            break
            
    if not found_fname or found_fname != function_name: 
        return False, f"Function name mismatch: expected {function_name}"
        
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'open']: return False, "Unsafe built-in"
            if node.func.id == 'NotImplementedError': return False, "Placeholder"
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module_name = getattr(node, 'module', None)
            for n in [n.name for n in node.names] + [module_name]:
                if n in ['os', 'sys', 'subprocess', 'requests', 'urllib', 'socket']: return False, "Unsafe import"
        if isinstance(node, ast.Pass): return False, "Placeholder pass"
    if 'humaneval' in code.lower(): return False, "HumanEval Contamination"
    if len(tests) < 3: return False, "Not enough tests generated"
    return execute_with_timeout(code, tests)

def format_prompt(bp):
    req = f"Write a complete, executable Python function named `{bp['function_name']}`.\n"
    req += f"It takes arguments: {', '.join(bp['argument_names'])}.\n"
    req += f"Domain: {bp['task_family']}. Operation: {bp['operation']}. Difficulty: {bp['difficulty']}.\n"
    req += f"Prompt style: {bp['prompt_format']}.\n"
    req += f"CRITICAL Edge cases to handle: {bp['required_edge_cases']}.\n"
    req += "Ensure the output code contains NO markdown formatting, NO prose, NO backticks, just the exact python function starting with `def `.\n"
    req += "CRITICAL: The solution MUST contain a docstring with at least 3 doctest examples (>>>).\n"
    return req

def run_pipeline(target_holdout=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Qwen...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    with open("data/stage6A_train_blueprints.jsonl") as f:
        all_train_bps = [json.loads(l) for l in f]
    
    # Skip the first 1500 which were used for train!
    # To be extremely safe, we read the existing stage6A_blueprint_train.jsonl to find out how many were used
    used_bps = set()
    with open("data/stage6A_blueprint_train.jsonl") as f:
        for l in f:
            used_bps.add(json.loads(l)["id"])
            
    available_bps = [bp for bp in all_train_bps if bp["id"] not in used_bps]
    print(f"Skipped {len(used_bps)} used blueprints. {len(available_bps)} available for natural holdout.")
    
    out_holdout = "data/stage6A_blueprint_natural_holdout.jsonl"
    if os.path.exists(out_holdout): os.remove(out_holdout)
    f_holdout = open(out_holdout, "a", encoding="utf-8")
    
    holdout_count = 0
    total_attempted = 0
    rejected_count = 0
    rejection_reasons = defaultdict(int)
    seen_bodies = set()
    pbar = tqdm(total=target_holdout)
    bp_idx = 0
    
    while holdout_count < target_holdout:
        batch_bps = []
        for _ in range(BATCH_SIZE):
            if bp_idx < len(available_bps) and holdout_count + len(batch_bps) < target_holdout:
                batch_bps.append(available_bps[bp_idx])
                bp_idx += 1
        
        if not batch_bps:
            print("Ran out of blueprints!")
            break
            
        texts = []
        for bp in batch_bps:
            messages = [
                {"role": "system", "content": "You are a senior python engineer writing self-contained functions. Output ONLY pure python code."},
                {"role": "user", "content": format_prompt(bp)}
            ]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            
        for j, bp in enumerate(batch_bps):
            if holdout_count >= target_holdout: break
            total_attempted += 1
            gen_ids = outputs[j][len(inputs.input_ids[j]):]
            code = tok.decode(gen_ids, skip_special_tokens=True).strip()
            code = strip_fences(code)
            
            fname = bp["function_name"]
            
            tests = extract_doctests(code)
            all_tests = tests + bp["hidden_tests"]
            
            is_valid, reason = validate_task(code, all_tests, fname)
            if is_valid:
                prompt_split, body_split = split_function(code)
                if not prompt_split or not body_split:
                    rejection_reasons["Failed to split function"] += 1
                    rejected_count += 1
                    continue
                    
                body_hash = hash(body_split)
                if body_hash in seen_bodies:
                    rejection_reasons["Duplicate code body"] += 1
                    rejected_count += 1
                    continue
                seen_bodies.add(body_hash)
                    
                ex = {
                    "id": bp["id"] + "_nat_hld",
                    "prompt": prompt_split,
                    "target_completion": body_split,
                    "target_code_full": code,
                    "function_name": fname,
                    "task_type": bp["task_family"],
                    "tests": tests,
                    "hidden_tests": bp["hidden_tests"],
                    "difficulty": bp["difficulty"],
                    "validation_status": "Valid",
                    "source": "stage6_blueprint_natural_holdout",
                    "teacher_model": "Qwen2.5-Coder-3B-Instruct",
                    "prompt_format": bp["prompt_format"]
                }
                f_holdout.write(json.dumps(ex) + "\n")
                holdout_count += 1
                pbar.update(1)
            else:
                rejection_reasons[reason] += 1
                rejected_count += 1

        print(f"Batch done. Natural Holdout={holdout_count}/{target_holdout}", flush=True)

    f_holdout.close()
    pbar.close()
    print("Done generating natural holdout!")

if __name__ == "__main__":
    run_pipeline(target_holdout=100)
