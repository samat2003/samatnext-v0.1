import sys, json, torch, os, random, ast, uuid
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 32

TASK_FAMILIES = ["strings", "lists", "dictionaries", "math", "recursion-lite", "sorting/searching", "simple algorithms", "edge-case handling", "data transformation"]
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

def execute_with_timeout(code, tests):
    exec_globals = {}
    try:
        # User requested no eval/exec inside the target code, but we must use exec to validate the unit tests
        full_code = code + "\n" + "\n".join(tests)
        exec(full_code, exec_globals)
        return True
    except Exception as e:
        return False

def validate_task(code, tests, function_name):
    # 1. ast.parse passes
    try:
        parsed = ast.parse(code)
    except:
        return False, "Syntax error"
        
    # 2. function name exists
    func_names = [node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
    if function_name not in func_names:
        return False, "Function name mismatch"
        
    # 3. no unsafe code
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'open', 'input', 'print']:
                    return False, "Unsafe or forbidden built-in"
                if node.func.id == 'NotImplementedError':
                    return False, "Placeholder"
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for n in node.names:
                if n.name in ['os', 'sys', 'subprocess', 'requests', 'urllib', 'socket']:
                    return False, "Unsafe import"
        if isinstance(node, ast.Pass):
            return False, "Placeholder pass"
            
    # 4. no HumanEval contamination
    code_lower = code.lower()
    if 'humaneval' in code_lower:
        return False, "Contamination"
        
    # 5. execute tests
    if not execute_with_timeout(code, tests):
        return False, "Unit tests failed"
        
    return True, "Valid"

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
        if is_holdout:
            req += "Ensure highly obscure function names and variable names (e.g., prefix with hld_ or use uncommon synonyms). Do not use common names like add, foo, x, y.\n"
            
        req += """Output exactly in this format:
[FUNCTION_NAME]
the_function_name
[PROMPT]
The prompt text
[SOLUTION]
def the_function_name(...):
...
[TESTS]
assert the_function_name(...) == ...
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
    target_train = 30000
    target_holdout = 1000
    
    out_train = os.path.join("data", "stage6_humaneval_style_train.jsonl")
    out_holdout = os.path.join("data", "stage6_humaneval_style_holdout.jsonl")
    
    os.makedirs("data", exist_ok=True)
    
    # Load existing to resume
    train_count = 0
    holdout_count = 0
    if os.path.exists(out_train):
        train_count = sum(1 for _ in open(out_train))
    if os.path.exists(out_holdout):
        holdout_count = sum(1 for _ in open(out_holdout))
        
    print(f"Existing: Train={train_count}/{target_train}, Holdout={holdout_count}/{target_holdout}")
    
    if train_count >= target_train and holdout_count >= target_holdout:
        print("Done!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    f_train = open(out_train, "a")
    f_holdout = open(out_holdout, "a")
    
    # Pre-generate batches
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
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            
        for j, inst in enumerate(batch_instructions):
            gen_ids = outputs[j][len(inputs.input_ids[j]):]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            
            sections = extract_sections(gen_text)
            
            if "FUNCTION_NAME" in sections and "PROMPT" in sections and "SOLUTION" in sections and "TESTS" in sections:
                fname = sections["FUNCTION_NAME"].split()[0].strip()
                code = sections["SOLUTION"]
                tests = [t for t in sections["TESTS"].split('\n') if t.strip().startswith('assert')]
                
                is_valid, reason = validate_task(code, tests, fname)
                if is_valid:
                    ex = {
                        "id": str(uuid.uuid4()),
                        "prompt": sections["PROMPT"],
                        "target_code": code,
                        "function_name": fname,
                        "task_type": inst["task_type"],
                        "tests": tests,
                        "difficulty": inst["difficulty"],
                        "validation_status": reason
                    }
                    if inst["is_holdout"] and holdout_count < target_holdout:
                        f_holdout.write(json.dumps(ex) + "\n")
                        f_holdout.flush()
                        holdout_count += 1
                    elif not inst["is_holdout"] and train_count < target_train:
                        f_train.write(json.dumps(ex) + "\n")
                        f_train.flush()
                        train_count += 1
                        
        print(f"Progress: Train={train_count}/{target_train}, Holdout={holdout_count}/{target_holdout}")

    f_train.close()
    f_holdout.close()

if __name__ == "__main__":
    main()
