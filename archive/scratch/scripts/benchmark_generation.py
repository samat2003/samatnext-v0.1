import sys, json, torch, os, random, ast, uuid, subprocess, tempfile, re, time
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

TASK_FAMILIES = ["strings", "lists", "dictionaries", "math", "recursion-lite", "sorting/searching", "simple algorithms", "edge-case handling", "data transformation"]
PROMPT_FORMATS = ["HumanEval-style function signature + docstring", "natural language function request", "examples-only prompt", "test-driven prompt", "bug-fix prompt", "partial-code completion prompt", "edge-case-heavy prompt"]
DIFFICULTIES = ["easy", "medium", "hard"]

def get_words(text): return set(text.lower().split())
def is_near_duplicate(w1, w2): return len(w1.intersection(w2)) / len(w1.union(w2)) > 0.8

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

def generate_hidden_tests(fname, code, prompt):
    tests = []; fname_lower = fname.lower(); prompt_lower = prompt.lower()
    if "factorial" in fname_lower: tests.extend([f"assert {fname}(0) == 1", f"assert {fname}(1) == 1"])
    elif "safe" in fname_lower and "divid" in fname_lower and "none" in prompt_lower: tests.append(f"assert {fname}(1, 0) is None")
    elif "first" in fname_lower and "item" in fname_lower: tests.append(f"try:\n    {fname}([])\nexcept:\n    pass")
    elif "remove" in fname_lower and "duplicate" in fname_lower: tests.append(f"assert {fname}([]) == []")
    elif "palindrome" in fname_lower: tests.extend([f"assert {fname}('') == True", f"assert {fname}('a') == True"])
    elif "merge" in fname_lower and "sort" in fname_lower: tests.extend([f"assert {fname}([], []) == []", f"assert {fname}([1], []) == [1]"])
    elif "dict" in fname_lower and "get" in fname_lower: tests.append(f"assert {fname}({{}}, 'a', 'default') == 'default'")
    elif "normalize" in fname_lower and "string" in fname_lower: tests.append(f"assert {fname}('') == ''")
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
    if function_name == "unknown": return False, "Function name mismatch"
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

def extract_sections(text):
    sections = {}; current_key = None; current_content = []
    for line in text.split('\n'):
        if line.startswith('[') and line.endswith(']'):
            if current_key: sections[current_key] = '\n'.join(current_content).strip()
            current_key = line[1:-1]; current_content = []
        else:
            if current_key: current_content.append(line)
    if current_key: sections[current_key] = '\n'.join(current_content).strip()
    return sections

def generate_instructions(num_tasks):
    instructions = []
    for _ in range(num_tasks):
        family = random.choice(TASK_FAMILIES)
        fmt = random.choice(PROMPT_FORMATS)
        diff = random.choice(DIFFICULTIES)
        req = f"Create a new, unique Python coding task. Domain: {family}. Difficulty: {diff}. Prompt format: {fmt}.\n"
        req += f"Ensure the task scenario is highly distinct from typical examples. Random seed constraint: {random.randint(10000, 99999)}.\n"
        req += "Ensure the output code contains NO markdown formatting, NO prose, NO backticks, just the code itself.\n"
        req += "CRITICAL: The solution MUST contain a docstring with at least 3 doctest examples (>>>).\n"
        req += "Output exactly in this format:\n[PROMPT]\nThe exact prompt text to give to the model.\n\n[SOLUTION]\ndef the_function_name(...):\n    \"\"\"\n    >>> the_function_name(...)\n    ...\n    \"\"\"\n    ...\n"
        instructions.append({"task_type": family, "difficulty": diff, "format": fmt, "instruction": req})
    return instructions

def run_benchmark(model, tok, device, batch_size, total_attempts=100):
    print(f"\n{'='*50}\nBenchmarking Batch Size: {batch_size}\n{'='*50}")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    accepted_examples = []
    rejected_count = 0
    generated_tokens = []
    
    start_time = time.time()
    
    attempts = 0
    oom_crash = False
    
    while attempts < total_attempts:
        current_batch_size = min(batch_size, total_attempts - attempts)
        batch_instructions = generate_instructions(current_batch_size)
        texts = []
        for inst in batch_instructions:
            messages = [
                {"role": "system", "content": "You are a dataset generator for Python coding tasks. Always output the exact requested bracket format."},
                {"role": "user", "content": inst["instruction"]}
            ]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        except RuntimeError as e:
            if "out of memory" in str(e):
                oom_crash = True
                print("OOM CRASH DETECTED!")
                break
            else:
                raise e
                
        attempts += current_batch_size
        
        for j, inst in enumerate(batch_instructions):
            gen_ids = outputs[j][len(inputs.input_ids[j]):]
            generated_tokens.append(len(gen_ids))
            gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            sections = extract_sections(gen_text)
            
            if "PROMPT" in sections and "SOLUTION" in sections:
                code = strip_fences(sections["SOLUTION"])
                prompt = sections["PROMPT"]
                try:
                    parsed = ast.parse(code)
                    func_names = [node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
                    fname = func_names[0] if func_names else "unknown"
                except: fname = "unknown"
                
                tests = extract_doctests(code)
                hidden_tests = generate_hidden_tests(fname, code, prompt)
                all_tests = tests + hidden_tests
                
                is_valid, reason = validate_task(code, all_tests, fname)
                if is_valid:
                    prompt_split, body_split = split_function(code)
                    if prompt_split and body_split:
                        accepted_examples.append({
                            "function_name": fname,
                            "target_completion": body_split
                        })
                    else:
                        rejected_count += 1
                else:
                    rejected_count += 1
            else:
                rejected_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0
    
    if oom_crash:
        return {
            "batch_size": batch_size,
            "status": "OOM/CRASH"
        }
        
    accepted_count = len(accepted_examples)
    acc_rate = accepted_count / total_attempts if total_attempts > 0 else 0
    time_per_attempt = total_time / total_attempts if total_attempts > 0 else 0
    time_per_accept = total_time / accepted_count if accepted_count > 0 else float('inf')
    avg_tokens = sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0
    
    return {
        "batch_size": batch_size,
        "status": "SUCCESS",
        "attempted": total_attempts,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "acc_rate": acc_rate,
        "total_time": total_time,
        "time_per_attempt": time_per_attempt,
        "time_per_accept": time_per_accept,
        "peak_vram": peak_vram,
        "avg_tokens": avg_tokens,
        "sample_examples": [e["function_name"] for e in accepted_examples[:3]]
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    batch_sizes = [4, 8, 12, 16]
    results = []
    
    for b in batch_sizes:
        res = run_benchmark(model, tok, device, b, total_attempts=100)
        results.append(res)
        if res["status"] == "OOM/CRASH":
            break
            
    print("\n\n" + "="*80)
    print("BENCHMARK REPORT")
    print("="*80)
    
    for r in results:
        print(f"\nBatch Size: {r['batch_size']} | Status: {r['status']}")
        if r['status'] == "SUCCESS":
            print(f"Attempted: {r['attempted']} | Accepted: {r['accepted']} | Rejected: {r['rejected']}")
            print(f"Acceptance Rate: {r['acc_rate']:.1%}")
            print(f"Time per attempt: {r['time_per_attempt']:.2f}s")
            print(f"Time per accepted: {r['time_per_accept']:.2f}s")
            print(f"Peak VRAM: {r['peak_vram']:.2f} GB")
            print(f"Avg Generated Tokens: {r['avg_tokens']:.1f}")
            print(f"Sample Accepted Functions: {r['sample_examples']}")
    print("="*80)

if __name__ == "__main__":
    main()
