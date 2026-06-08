import os, sys, json, ast, re, random
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NUM_TEST_SAMPLES = 100
OUT_FILE = os.path.join(ROOT, "data", "stage5_teacher_distill_test.jsonl")

# --- STRICT SEMANTIC TEMPLATES ---

TASKS = [
    # 1. Simple Function
    {
        "type": "simple_function",
        "name": "add_numbers",
        "alt_names": ["sum_two", "calculate_sum", "add_vals"],
        "prompts": [
            "Write a Python function {func_name}(a, b) that returns the sum of a and b.",
            "Write {func_name}(a, b). It should return the sum of the two inputs."
        ],
        "no_name_prompts": [
            "Write a Python function that takes two arguments and returns their sum.",
            "I need a short Python function that adds two numbers together."
        ],
        "tests": ["assert {func_name}(2, 3) == 5", "assert {func_name}(-1, 1) == 0"]
    },
    {
        "type": "simple_function",
        "name": "square_number",
        "alt_names": ["get_square", "calc_square", "square_val"],
        "prompts": [
            "Implement {func_name}(x) in Python to return the square of x.",
            "Write {func_name}(n). Return n squared."
        ],
        "no_name_prompts": [
            "Implement a function in Python to return the square of its input.",
            "Write a function that squares a number."
        ],
        "tests": ["assert {func_name}(2) == 4", "assert {func_name}(-3) == 9"]
    },
    # 2. List/String/Dict
    {
        "type": "list_string_dict",
        "name": "remove_duplicates",
        "alt_names": ["dedupe_list", "unique_items", "clean_duplicates"],
        "prompts": [
            "Create a function {func_name}(items) that returns a new list with duplicates removed while preserving order.",
            "Implement {func_name}(elements). Return a new list containing unique elements in their original order."
        ],
        "no_name_prompts": [
            "Create a function that returns a new list with duplicates removed while preserving order."
        ],
        "tests": ["assert {func_name}([1, 2, 2, 3, 1]) == [1, 2, 3]", "assert {func_name}([]) == []"]
    },
    {
        "type": "list_string_dict",
        "name": "normalize_text",
        "alt_names": ["clean_string", "format_text", "strip_lower"],
        "prompts": [
            "Implement {func_name}(text). It should strip leading/trailing spaces and convert the result to lowercase.",
            "Write {func_name}(s) to clean a string by trimming whitespace and making it lowercase."
        ],
        "no_name_prompts": [
            "Implement a function that strips leading/trailing spaces and converts the result to lowercase."
        ],
        "tests": ["assert {func_name}('  Hello World  ') == 'hello world'"]
    },
    {
        "type": "list_string_dict",
        "name": "safe_get",
        "alt_names": ["dict_lookup", "get_with_default", "fetch_key"],
        "prompts": [
            "Write {func_name}(mapping, key, default=None), returning mapping[key] when the key exists and default otherwise."
        ],
        "no_name_prompts": [
            "Write a function that takes a dictionary, a key, and an optional default. It should return the value if the key exists, and the default otherwise."
        ],
        "tests": ["assert {func_name}({'a': 1}, 'a') == 1", "assert {func_name}({'a': 1}, 'b', 0) == 0"]
    },
    # 3. Edge Cases
    {
        "type": "edge_case",
        "name": "first_or_none",
        "alt_names": ["get_first", "safe_first", "head_element"],
        "prompts": [
            "Define {func_name}(items). Return the first item in the list, or None if the list is empty."
        ],
        "no_name_prompts": [
            "Define a function that returns the first item in a list, or None if the list is empty."
        ],
        "tests": ["assert {func_name}([1, 2]) == 1", "assert {func_name}([]) is None"]
    },
    {
        "type": "edge_case",
        "name": "divide_or_zero",
        "alt_names": ["safe_divide", "calc_ratio", "divide_vals"],
        "prompts": [
            "Write {func_name}(a, b). Return a / b, but return 0 if b is 0."
        ],
        "no_name_prompts": [
            "Write a function to divide a by b, returning 0 if b is 0."
        ],
        "tests": ["assert {func_name}(10, 2) == 5.0", "assert {func_name}(10, 0) == 0"]
    },
    # 4. Bug Fixes
    {
        "type": "bug_fix",
        "name": "is_even",
        "alt_names": ["check_even", "is_even_number"],
        "prompts": [
            "Fix {func_name}(n) so it returns True only when n is even:\ndef {func_name}(n):\n    return n % 2 == 1"
        ],
        "no_name_prompts": [
            "Fix this function so it returns True only when n is even:\ndef is_even(n):\n    return n % 2 == 1"
        ],
        "tests": ["assert {func_name}(2) is True", "assert {func_name}(3) is False"],
        "enforce_name": "is_even" # if no name provided in prompt template, this is what it's named in the bad code
    },
    {
        "type": "bug_fix",
        "name": "sum_numbers",
        "alt_names": ["sum_list", "total_items"],
        "prompts": [
            "Fix {func_name}(items) so it returns the sum of the list:\ndef {func_name}(items):\n    return 0"
        ],
        "no_name_prompts": [
            "Fix sum_numbers(items) so it returns the sum of the list:\ndef sum_numbers(items):\n    return 0"
        ],
        "tests": ["assert {func_name}([1, 2, 3]) == 6", "assert {func_name}([]) == 0"],
        "enforce_name": "sum_numbers"
    },
    # 5. Algorithm
    {
        "type": "algorithm",
        "name": "factorial",
        "alt_names": ["calc_factorial", "get_factorial"],
        "prompts": [
            "Write {func_name}(n). Return 1 when n is 0. Raise ValueError if n is negative. Otherwise return factorial."
        ],
        "no_name_prompts": [
            "Write a function to calculate factorial. Return 1 when n is 0. Raise ValueError if n is negative."
        ],
        "tests": [
            "assert {func_name}(3) == 6", 
            "assert {func_name}(0) == 1",
            "try:\n    {func_name}(-1)\n    assert False\nexcept ValueError:\n    pass"
        ]
    },
    {
        "type": "algorithm",
        "name": "is_palindrome",
        "alt_names": ["check_palindrome", "palindrome_str"],
        "prompts": [
            "Write {func_name}(s) to check if a string is a palindrome, ignoring spaces."
        ],
        "no_name_prompts": [
            "Write a function to check if a string is a palindrome, ignoring spaces."
        ],
        "tests": ["assert {func_name}('race car') is True", "assert {func_name}('hello') is False"]
    },
    # 6. Test Driven
    {
        "type": "test_driven",
        "name": "merge_sorted",
        "alt_names": ["combine_sorted", "merge_lists"],
        "prompts": [
            "Write {func_name}(a, b) so that these pass:\nassert {func_name}([1,3], [2,4]) == [1,2,3,4]\nassert {func_name}([], [1]) == [1]"
        ],
        "no_name_prompts": [
            "Write a function called merge_sorted(a, b) so that these pass:\nassert merge_sorted([1,3], [2,4]) == [1,2,3,4]\nassert merge_sorted([], [1]) == [1]"
        ],
        "tests": ["assert {func_name}([1,3], [2,4]) == [1,2,3,4]", "assert {func_name}([], [1]) == [1]"],
        "enforce_name": "merge_sorted"
    },
    # 7. Refactor
    {
        "type": "refactor",
        "name": "remove_duplicates",
        "alt_names": ["dedupe_items", "clean_list"],
        "prompts": [
            "Refactor {func_name}(items) clearly without changing behavior. It should remove duplicates while preserving order:\ndef {func_name}(items):\n    r=[]\n    for i in items:\n        if i not in r:r.append(i)\n    return r"
        ],
        "no_name_prompts": [
            "Refactor this function clearly without changing behavior. It should remove duplicates while preserving order:\ndef clean_items(items):\n    r=[]\n    for i in items:\n        if i not in r:r.append(i)\n    return r"
        ],
        "tests": ["assert {func_name}([1, 2, 2, 3, 1]) == [1, 2, 3]", "assert {func_name}([]) == []"],
        "enforce_name": "clean_items"
    }
]

def generate_prompts(num):
    prompts = []
    random.seed(42)
    for i in range(num):
        task = random.choice(TASKS)
        
        has_name = random.random() < 0.60
        if has_name:
            fn = random.choice([task["name"]] + task["alt_names"])
            p = random.choice(task["prompts"]).replace("{func_name}", fn)
        else:
            if "enforce_name" in task:
                fn = task["enforce_name"]
            else:
                fn = None
            p = random.choice(task["no_name_prompts"])
            
        prompts.append({
            "id": f"stage5_test_{i:06d}",
            "prompt": p,
            "function_name": fn,
            "task_type": task["type"],
            "has_name": has_name,
            "tests": task["tests"]
        })
    return prompts

def extract_fn(code):
    m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code)
    return m.group(1) if m else None

def validate(ex, code):
    reasons = []
    if not code.strip():
        return "rejected", ["Empty output"]
        
    if "```" in code:
        reasons.append("Markdown fences")
        
    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError:
        reasons.append("Syntax error")
        return "rejected", reasons

    # Function name check
    gen_fn = extract_fn(code)
    if not gen_fn:
        reasons.append("No function definition found")
        return "rejected", reasons

    if ex["function_name"]:
        if gen_fn != ex["function_name"]:
            reasons.append(f"Wrong function name: got '{gen_fn}', expected '{ex['function_name']}'")
            
    # Unsafe / prohibited
    if any(x in code for x in ["import os", "import subprocess", "eval(", "exec(", "input(", "print("]):
        reasons.append("Prohibited calls/imports")
        
    if any(x in code for x in ["pass", "TODO", "NotImplemented"]):
        if ex["task_type"] != "bug_fix":
            reasons.append("Placeholder code")

    # Semantic Test Execution
    fn_to_test = ex["function_name"] if ex["function_name"] else gen_fn
    ns = {}
    try:
        exec(compile(code, "<string>", "exec"), ns)
    except Exception as e:
        reasons.append(f"Execution error: {e}")
        return "rejected", reasons
        
    for t in ex["tests"]:
        t_eval = t.replace("{func_name}", fn_to_test)
        try:
            exec(compile(t_eval, "<string>", "exec"), ns)
        except Exception as e:
            reasons.append(f"Test failed: {t_eval} ({type(e).__name__})")

    if reasons:
        return "rejected", reasons
    return "passed", []

def main():
    print(f"Generating {NUM_TEST_SAMPLES} test prompts...")
    prompts = generate_prompts(NUM_TEST_SAMPLES)
    
    with open("data/stage5_test_prompts.jsonl", "w") as f:
        for p in prompts: f.write(json.dumps(p) + "\n")
        
    print("Running batch generation (Qwen2.5-Coder-3B-Instruct)...")
    subprocess.run(["python", "scripts/run_teacher_batch.py", "data/stage5_test_prompts.jsonl", "data/stage5_test_outputs.jsonl"], check=True)
    
    outputs = [json.loads(l) for l in open("data/stage5_test_outputs.jsonl")]
    accepted, rejected = [], []
    
    digit_suffix_count = 0
    malformed_prompt_count = 0
    
    for ex in outputs:
        # Check constraints requested by user for diagnostics
        if ex["function_name"] and re.search(r"_\d+$", ex["function_name"]):
            digit_suffix_count += 1
        if "a function" in ex["prompt"] or "function(string)" in ex["prompt"]:
            malformed_prompt_count += 1
            
        status, reasons = validate(ex, ex["teacher_target"])
        ex["validation_status"] = status
        ex["rejection_reasons"] = reasons
        
        if status == "passed": accepted.append(ex)
        else: rejected.append(ex)
        
    print(f"\n=== Validation Stats ===")
    print(f"Total Generated: {len(outputs)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Acceptance Rate: {len(accepted)/len(outputs)*100:.1f}%")
    print(f"Test Pass Rate: {len(accepted)/len(outputs)*100:.1f}%")  # Tests are enforced strictly in validation
    print(f"Malformed Prompts: {malformed_prompt_count}")
    print(f"Digit-Suffix Functions: {digit_suffix_count}")
    
    with open(OUT_FILE, "w") as f:
        for a in accepted: f.write(json.dumps(a) + "\n")
        
    print("\n--- 20 ACCEPTED EXAMPLES ---")
    for a in accepted[:20]:
        print(f"\n[{a['task_type'].upper()}] Prompt:\n{a['prompt']}\nTarget:\n{a['teacher_target'][:120]}...\n")
        
    print("\n--- 20 REJECTED EXAMPLES ---")
    for r in rejected[:20]:
        print(f"\n[{r['task_type'].upper()}] Prompt:\n{r['prompt']}")
        print(f"Reasons: {r['rejection_reasons']}")
        print(f"Target:\n{r['teacher_target'][:120]}...\n")

if __name__ == "__main__":
    main()
