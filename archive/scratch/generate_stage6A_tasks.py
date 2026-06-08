import sys, json, torch, os, random, ast, uuid, subprocess, tempfile, re
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 16

TASK_FAMILIES = [
    "strings", "lists", "dictionaries", "math", "recursion-lite", 
    "sorting/searching", "simple algorithms", "edge-case handling", "data transformation"
]
PROMPT_FORMATS = [
    "HumanEval-style function signature + docstring",
    "natural language function request",
    "examples-only prompt",
    "test-driven prompt",
    "bug-fix prompt",
    "partial-code completion prompt",
    "edge-case-heavy prompt"
]
DIFFICULTIES = ["easy", "medium", "hard"]

def get_words(text):
    return set(text.lower().split())

def is_near_duplicate(w1, w2):
    if not w1 or not w2: return False
    return len(w1.intersection(w2)) / len(w1.union(w2)) > 0.8

def extract_doctests(code):
    tests = []
    lines = code.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>>> '):
            expr = line[4:].strip()
            j = i + 1
            expected = []
            while j < len(lines) and not lines[j].strip().startswith('>>>') and lines[j].strip() not in ('"""', "'''") and not lines[j].strip().startswith('def '):
                if lines[j].strip():
                    expected.append(lines[j].strip())
                j += 1
            if expected:
                exp_str = '\n'.join(expected)
                tests.append(f"assert {expr} == {exp_str}")
            i = j
        else:
            i += 1
    return tests

def generate_hidden_tests(fname, code, prompt):
    tests = []
    fname_lower = fname.lower()
    prompt_lower = prompt.lower()
    
    if "factorial" in fname_lower:
        tests.append(f"assert {fname}(0) == 1")
        tests.append(f"assert {fname}(1) == 1")
    elif "safe" in fname_lower and "divid" in fname_lower:
        if "none" in prompt_lower:
            tests.append(f"assert {fname}(1, 0) is None")
    elif "first" in fname_lower and "item" in fname_lower:
        tests.append(f"try:\n    {fname}([])\nexcept:\n    pass")
    elif "remove" in fname_lower and "duplicate" in fname_lower:
        tests.append(f"assert {fname}([]) == []")
    elif "palindrome" in fname_lower:
        tests.append(f"assert {fname}('') == True")
        tests.append(f"assert {fname}('a') == True")
    elif "merge" in fname_lower and "sort" in fname_lower:
        tests.append(f"assert {fname}([], []) == []")
        tests.append(f"assert {fname}([1], []) == [1]")
    elif "dict" in fname_lower and "get" in fname_lower:
        tests.append(f"assert {fname}({{}}, 'a', 'default') == 'default'")
    elif "normalize" in fname_lower and "string" in fname_lower:
        tests.append(f"assert {fname}('') == ''")
        
    return tests

def execute_with_timeout(code, tests):
    test_code = code + "\n" + "\n".join(tests)
    script_content = f"""
import math, collections, itertools, functools, re, string
{test_code}
print('__SUCCESS__')
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        temp_path = f.name
        
    try:
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=2.0)
        if result.returncode == 0 and "__SUCCESS__" in result.stdout:
            return True, "Valid"
        else:
            return False, f"Test failed: {result.stderr.strip()[:100]}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:100]
    finally:
        os.remove(temp_path)

def strip_fences(code):
    match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()

def validate_task(code, tests, function_name):
    code = strip_fences(code)
    try:
        parsed = ast.parse(code)
    except:
        return False, "Syntax error"
        
    if function_name == "unknown":
        return False, "Function name mismatch"
        
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'open']:
                    return False, "Unsafe built-in"
                if node.func.id == 'NotImplementedError':
                    return False, "Placeholder"
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module_name = getattr(node, 'module', None)
            names = [n.name for n in node.names]
            for n in names + [module_name]:
                if n in ['os', 'sys', 'subprocess', 'requests', 'urllib', 'socket']:
                    return False, "Unsafe import"
        if isinstance(node, ast.Pass):
            return False, "Placeholder pass"
            
    code_lower = code.lower()
    if 'humaneval' in code_lower:
        return False, "HumanEval Contamination"
        
    if len(tests) < 3:
        return False, "Not enough tests generated"
        
    success, exec_reason = execute_with_timeout(code, tests)
    if not success:
        return False, exec_reason
        
    return True, "Valid"

def split_function(code):
    try:
        tree = ast.parse(code)
    except:
        return None, None
        
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return None, None
        
    func_def = tree.body[0]
    lines = code.split("\n")
    
    if ast.get_docstring(func_def) and isinstance(func_def.body[0], ast.Expr) and isinstance(func_def.body[0].value, ast.Constant):
        doc_node = func_def.body[0]
        split_line = doc_node.end_lineno
    else:
        first_stmt = func_def.body[0]
        split_line = first_stmt.lineno - 1
        
    prompt = "\n".join(lines[:split_line]) + "\n"
    body = "\n".join(lines[split_line:])
    
    return prompt, body

def extract_sections(text):
    sections = {}
    current_key = None
    current_content = []
    
    for line in text.split('\n'):
        if line.startswith('[') and line.endswith(']'):
            if current_key:
                sections[current_key] = '\n'.join(current_content).strip()
            current_key = line[1:-1]
            current_content = []
        else:
            if current_key:
                current_content.append(line)
                
    if current_key:
        sections[current_key] = '\n'.join(current_content).strip()
        
    return sections

def generate_instructions(num_tasks, is_holdout=False):
    instructions = []
    for _ in range(num_tasks):
        family = random.choice(TASK_FAMILIES)
        fmt = random.choice(PROMPT_FORMATS)
        diff = random.choice(DIFFICULTIES)
        
        req = f"Create a new, unique Python coding task. Domain: {family}. Difficulty: {diff}. Prompt format: {fmt}.\n"
        req += f"Ensure the task scenario is highly distinct from typical examples. Random seed constraint: {random.randint(10000, 99999)}.\n"
        req += "Ensure the output code contains NO markdown formatting, NO prose, NO backticks, just the code itself.\n"
        if is_holdout:
            req += "CRITICAL: Ensure highly obscure function names and variable names (e.g., prefix with hld_ or use uncommon synonyms). Do not use common names like add, foo, x, y, calculate, sort, filter.\n"
            
        req += "CRITICAL: The solution MUST contain a docstring with at least 3 doctest examples (>>>).\n"
        req += """Output exactly in this format:
[PROMPT]
The exact prompt text to give to the model.

[SOLUTION]
def the_function_name(...):
    \"\"\"
    >>> the_function_name(...)
    ...
    \"\"\"
    ...
"""
        instructions.append({
            "task_type": family,
            "difficulty": diff,
            "format": fmt,
            "instruction": req,
            "is_holdout": is_holdout
        })
    return instructions

def main():
    target_train = 5000
    target_holdout = 500
    
    out_train = os.path.join("data", "stage6A_train.jsonl")
    out_holdout = os.path.join("data", "stage6A_holdout.jsonl")
    
    os.makedirs("data", exist_ok=True)
    
    train_count = 0
    holdout_count = 0
    total_attempted = 0
    rejected_count = 0
    rejection_reasons = defaultdict(int)
    family_dist = defaultdict(int)
    format_dist = defaultdict(int)
    diff_dist = defaultdict(int)
    function_names = set()
    holdout_function_names = set()
    train_function_names = set()
    duplicate_names = 0
    unsafe_code_count = 0
    
    prompt_lengths = []
    code_lengths = []
    tasks_with_hidden = 0
    fname_counter = Counter()
    
    accepted_examples = []
    rejected_examples = []
    accepted_prompt_words = []
    near_duplicate_prompts = 0
    
    if os.path.exists(out_train):
        os.remove(out_train)
    if os.path.exists(out_holdout):
        os.remove(out_holdout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    f_train = open(out_train, "a")
    f_holdout = open(out_holdout, "a")
    
    pbar = tqdm(total=target_train + target_holdout)
    
    while train_count < target_train or holdout_count < target_holdout:
        batch_instructions = []
        for _ in range(BATCH_SIZE):
            if holdout_count < target_holdout:
                batch_instructions.append(generate_instructions(1, is_holdout=True)[0])
            else:
                batch_instructions.append(generate_instructions(1, is_holdout=False)[0])
                
        texts = []
        for inst in batch_instructions:
            messages = [
                {"role": "system", "content": "You are a dataset generator for Python coding tasks. Always output the exact requested bracket format."},
                {"role": "user", "content": inst["instruction"]}
            ]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            
        for j, inst in enumerate(batch_instructions):
            total_attempted += 1
            gen_ids = outputs[j][len(inputs.input_ids[j]):]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            
            sections = extract_sections(gen_text)
            
            if "PROMPT" in sections and "SOLUTION" in sections:
                code = strip_fences(sections["SOLUTION"])
                prompt = sections["PROMPT"]
                try:
                    parsed = ast.parse(code)
                    func_names = [node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
                    fname = func_names[0] if func_names else "unknown"
                except:
                    fname = "unknown"
                    
                tests = extract_doctests(code)
                hidden_tests = generate_hidden_tests(fname, code, prompt)
                all_tests = tests + hidden_tests
                
                if fname in function_names:
                    duplicate_names += 1
                    rejection_reasons["Duplicate function name"] += 1
                    rejected_count += 1
                    if len(rejected_examples) < 30:
                        rejected_examples.append((fname, "Duplicate function name"))
                    continue
                    
                w_prompt = get_words(prompt)
                is_dup = any(is_near_duplicate(w_prompt, ex_w) for ex_w in accepted_prompt_words[-100:])
                if is_dup:
                    near_duplicate_prompts += 1
                    rejection_reasons["Near duplicate prompt"] += 1
                    rejected_count += 1
                    continue
                    
                is_valid, reason = validate_task(code, all_tests, fname)
                if is_valid:
                    prompt_split, body_split = split_function(code)
                    if not prompt_split or not body_split:
                        rejection_reasons["Failed to split function"] += 1
                        rejected_count += 1
                        if len(rejected_examples) < 30:
                            rejected_examples.append((fname, "Failed to split function"))
                        continue
                        
                    function_names.add(fname)
                    fname_counter[fname] += 1
                    prompt_lengths.append(len(prompt_split))
                    code_lengths.append(len(body_split))
                    accepted_prompt_words.append(w_prompt)
                    if hidden_tests:
                        tasks_with_hidden += 1
                        
                    ex = {
                        "id": str(uuid.uuid4()),
                        "prompt": prompt_split,
                        "target_completion": body_split,
                        "target_code_full": code,
                        "function_name": fname,
                        "task_type": inst["task_type"],
                        "tests": tests,
                        "hidden_tests": hidden_tests,
                        "difficulty": inst["difficulty"],
                        "validation_status": "Valid",
                        "source": "stage6_synthetic_teacher",
                        "teacher_model": "Qwen2.5-Coder-3B-Instruct",
                        "prompt_format": inst["format"]
                    }
                    if inst["is_holdout"] and holdout_count < target_holdout:
                        f_holdout.write(json.dumps(ex) + "\n")
                        holdout_count += 1
                        holdout_function_names.add(fname)
                        pbar.update(1)
                        if len(accepted_examples) < 30: accepted_examples.append(ex)
                    elif not inst["is_holdout"] and train_count < target_train:
                        f_train.write(json.dumps(ex) + "\n")
                        train_count += 1
                        train_function_names.add(fname)
                        pbar.update(1)
                        if len(accepted_examples) < 30: accepted_examples.append(ex)
                        
                    family_dist[inst["task_type"]] += 1
                    format_dist[inst["format"]] += 1
                    diff_dist[inst["difficulty"]] += 1
                else:
                    rejection_reasons[reason] += 1
                    rejected_count += 1
                    if "Unsafe" in reason: unsafe_code_count += 1
                    if len(rejected_examples) < 30:
                        rejected_examples.append((fname, reason))
            else:
                rejection_reasons["Format error (missing sections)"] += 1
                rejected_count += 1
                if len(rejected_examples) < 30:
                        rejected_examples.append(("N/A", "Format error"))

        print(f"Batch done. Progress: Train={train_count}/{target_train}, Holdout={holdout_count}/{target_holdout}", flush=True)
        print("Rejections:", dict(rejection_reasons), flush=True)

    f_train.close()
    f_holdout.close()
    pbar.close()
    
    overlap = len(train_function_names.intersection(holdout_function_names))
    avg_prompt = sum(prompt_lengths)/len(prompt_lengths) if prompt_lengths else 0
    avg_code = sum(code_lengths)/len(code_lengths) if code_lengths else 0
    perc_hidden = (tasks_with_hidden / (train_count + holdout_count)) * 100 if (train_count + holdout_count) > 0 else 0
    
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("STAGE 6A-MINI DATA VALIDATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"1. Total attempted: {total_attempted}")
    report_lines.append(f"2. Accepted train count: {train_count}")
    report_lines.append(f"3. Accepted holdout count: {holdout_count}")
    report_lines.append(f"4. Rejected count: {rejected_count}")
    report_lines.append("5. Rejection reasons:")
    for r, c in rejection_reasons.items():
        report_lines.append(f"   - {r}: {c}")
    report_lines.append("6. Task family distribution:")
    for r, c in family_dist.items(): report_lines.append(f"   - {r}: {c}")
    report_lines.append("7. Prompt format distribution:")
    for r, c in format_dist.items(): report_lines.append(f"   - {r}: {c}")
    report_lines.append("8. Difficulty distribution:")
    for r, c in diff_dist.items(): report_lines.append(f"   - {r}: {c}")
    report_lines.append(f"9. Duplicate function-name count: {duplicate_names}")
    report_lines.append(f"10. Train/holdout overlap count: {overlap}")
    report_lines.append(f"11. Unsafe-code rejection count: {unsafe_code_count}")
    report_lines.append(f"12. AST parse pass rate (accepted): 100.0%")
    report_lines.append(f"13. Unit test pass rate (accepted): 100.0%")
    
    report_lines.append("\n--- Additional Metrics ---")
    report_lines.append(f"Average prompt length: {avg_prompt:.1f} chars")
    report_lines.append(f"Average target_code length: {avg_code:.1f} chars")
    report_lines.append(f"Percent of tasks with hidden validator tests: {perc_hidden:.1f}%")
    report_lines.append(f"Near-duplicate prompt count: {near_duplicate_prompts}")
    report_lines.append("Top 20 most common function names:")
    for fname, cnt in fname_counter.most_common(20):
        report_lines.append(f"   - {fname}: {cnt}")
    
    report_lines.append("\n--- 30 Random Accepted Examples ---")
    for i, ex in enumerate(accepted_examples):
        ht = "Yes" if ex["hidden_tests"] else "No"
        report_lines.append(f"[{i+1}] {ex['function_name']} (Diff: {ex['difficulty']}, Family: {ex['task_type']}, Hidden Tests: {ht})")
        
    report_lines.append("\n--- 30 Random Rejected Examples ---")
    for i, (name, reason) in enumerate(rejected_examples):
        report_lines.append(f"[{i+1}] {name} -> {reason}")
    report_lines.append("=" * 60)
    
    report_text = "\\n".join(report_lines)
    print(report_text)
    
    os.makedirs("results", exist_ok=True)
    with open("results/stage6A_mini_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
if __name__ == "__main__":
    main()
