import os
import json
import random
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
EVAL_FILE = os.path.join(DATA_DIR, "stage2e_adversarial_holdout.jsonl")

NUM_EVAL = 300

def check_syntax(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def check_execution(code_str, tests):
    try:
        local_scope = {}
        exec(code_str, {}, local_scope)
        for test in tests:
            exec(test, {}, local_scope)
        return True
    except Exception:
        return False

# ==============================================================================
# NOVEL ADVERSARIAL TEMPLATES
# ==============================================================================

PREFIXES = [
    "I need a Python snippet that",
    "Can you implement",
    "Provide the code for",
    "Give me a function",
    "Please code a method",
    "Could you write a script",
    "Create a function for me",
    "Write a piece of code",
    "I require a Python function"
]

MID_1ARG = [
    "called {fn} accepting {a}",
    "named {fn} that processes {a}",
    "{fn}({a})",
    "where {fn} takes {a} as input"
]

MID_2ARG = [
    "called {fn} accepting {a} and {b}",
    "named {fn} that processes {a} and {b}",
    "{fn}({a}, {b})",
    "where {fn} takes {a} and {b} as input"
]

SUFFIX_RETURN = [
    "to compute and return",
    "which calculates",
    "that evaluates and returns",
    "to output",
    "that yields"
]

# NOVEL realistic argument name pools
P_2ARGS = [("left_val", "right_val"), ("first_num", "second_num"), ("input1", "input2")]
P_1ARG_NUM = [("input_num",), ("target_val",), ("n_val",), ("count_target",)]
P_1ARG_STR = [("input_string",), ("target_text",), ("source_str",)]
P_1ARG_LST = [("input_list",), ("source_array",), ("element_list",)]
P_2ARG_LST = [("list_a", "list_b"), ("array1", "array2")]
P_2ARG_MIX = [("source_dict", "target_key"), ("lookup_map", "key_val")]

TASKS = [
    {
        "category": "list_flattening", "task_type": "flatten_with_empty",
        "action": "a flattened list by one level from {a}, handling empty sublists.",
        "fn_names": ["flatten_nested", "flatten_complex", "reduce_list"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    res = []\n    for sub in {a}:\n        res.extend(sub)\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}([[1, 2], [], [3]]) == [1, 2, 3]", f"assert {fn}([]) == []", f"assert {fn}([[]]) == []"]
    },
    {
        "category": "safe_element", "task_type": "safe_first",
        "action": "the first element of {a}, returning None if it is totally empty.",
        "fn_names": ["grab_first", "first_safe", "safe_head"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return {a}[0] if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([10, 20]) == 10", f"assert {fn}([]) is None"]
    },
    {
        "category": "safe_element", "task_type": "safe_last",
        "action": "the last element of {a}, returning None if it is totally empty.",
        "fn_names": ["grab_last", "last_safe", "safe_tail"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return {a}[-1] if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([10, 20]) == 20", f"assert {fn}([]) is None"]
    },
    {
        "category": "string_normalization", "task_type": "string_normalize",
        "action": "the string {a} with leading/trailing whitespace removed and converted to lowercase.",
        "fn_names": ["normalize_text", "clean_string", "standardize_str"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return {a}.strip().lower()",
        "gen_tests": lambda fn: [f"assert {fn}('  HeLLo  ') == 'hello'", f"assert {fn}('') == ''", f"assert {fn}('TEST') == 'test'"]
    },
    {
        "category": "string_normalization", "task_type": "vowel_counter",
        "action": "the number of vowels in {a}, handling both uppercase and lowercase.",
        "fn_names": ["count_vowels_all", "tally_vowels", "find_all_vowels"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
        "gen_tests": lambda fn: [f"assert {fn}('AEIOU') == 5", f"assert {fn}('xyz') == 0", f"assert {fn}('aEiOu') == 5"]
    },
    {
        "category": "string_normalization", "task_type": "palindrome_complex",
        "action": "True if {a} is a palindrome ignoring spaces and case, else False.",
        "fn_names": ["is_complex_palindrome", "advanced_palindrome", "check_pal_hard"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    s = {a}.replace(' ', '').lower()\n    return s == s[::-1]",
        "gen_tests": lambda fn: [f"assert {fn}('Race Car') is True", f"assert {fn}('hello') is False", f"assert {fn}('A b A') is True"]
    },
    {
        "category": "remove_duplicates", "task_type": "dedup_preserve_order",
        "action": "a list with duplicate elements removed from {a} while preserving the original order.",
        "fn_names": ["unique_ordered", "dedup_stable", "remove_dups_keep_order"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    res = []\n    for x in {a}:\n        if x not in res:\n            res.append(x)\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}([3, 1, 2, 1, 3]) == [3, 1, 2]", f"assert {fn}([]) == []"]
    },
    {
        "category": "dict_operations", "task_type": "tally_dict",
        "action": "a dictionary with the frequency count of each item in {a}.",
        "fn_names": ["tally_up", "freq_map", "build_counts"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    c = {{}}\n    for x in {a}:\n        c[x] = c.get(x, 0) + 1\n    return c",
        "gen_tests": lambda fn: [f"assert {fn}(['a', 'b', 'a']) == {{'a': 2, 'b': 1}}", f"assert {fn}([]) == {{}}"]
    },
    {
        "category": "dict_operations", "task_type": "safe_dict_access",
        "action": "the value for {b} in {a}, but returns None if it doesn't exist.",
        "fn_names": ["secure_lookup", "safe_dict_get", "fetch_safe"],
        "arg_type": "2_mix", "code": "def {fn}({a}, {b}):\n    return {a}.get({b}, None)",
        "gen_tests": lambda fn: [f"assert {fn}({{'k': 1}}, 'k') == 1", f"assert {fn}({{}}, 'x') is None"]
    },
    {
        "category": "algorithms", "task_type": "fact_zero",
        "action": "the factorial of {a}, specifically ensuring 0 works correctly.",
        "fn_names": ["factorial_zero_safe", "safe_fact", "compute_fact_all"],
        "arg_type": 1, "code": "def {fn}({a}):\n    if {a} == 0:\n        return 1\n    res = 1\n    for i in range(1, {a} + 1):\n        res *= i\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}(0) == 1", f"assert {fn}(3) == 6", f"assert {fn}(5) == 120"]
    },
    {
        "category": "algorithms", "task_type": "fib_small",
        "action": "the {a}-th Fibonacci number (fib(0)=0, fib(1)=1).",
        "fn_names": ["fibonacci_base", "fib_core", "get_fib_num"],
        "arg_type": 1, "code": "def {fn}({a}):\n    if {a} <= 0: return 0\n    if {a} == 1: return 1\n    a_, b_ = 0, 1\n    for _ in range(2, {a} + 1):\n        a_, b_ = b_, a_ + b_\n    return b_",
        "gen_tests": lambda fn: [f"assert {fn}(0) == 0", f"assert {fn}(1) == 1", f"assert {fn}(2) == 1"]
    },
    {
        "category": "algorithms", "task_type": "custom_abs",
        "action": "the absolute value of {a} without using the built-in abs function.",
        "fn_names": ["absolute_custom", "my_abs", "abs_no_builtin"],
        "arg_type": 1, "code": "def {fn}({a}):\n    return -{a} if {a} < 0 else {a}",
        "gen_tests": lambda fn: [f"assert {fn}(-5) == 5", f"assert {fn}(5) == 5", f"assert {fn}(0) == 0"]
    },
    {
        "category": "algorithms", "task_type": "merge_sorted",
        "action": "a new list that merges {a} and {b} and sorts the result.",
        "fn_names": ["merge_and_sort", "sorted_merge", "combine_sorted"],
        "arg_type": "2_lst", "code": "def {fn}({a}, {b}):\n    return sorted({a} + {b})",
        "gen_tests": lambda fn: [f"assert {fn}([3, 1], [2, 4]) == [1, 2, 3, 4]", f"assert {fn}([], []) == []"]
    },
    {
        "category": "filters", "task_type": "filter_positives",
        "action": "a list containing only the positive numbers from {a}.",
        "fn_names": ["keep_positives", "filter_pos", "only_positives"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return [x for x in {a} if x > 0]",
        "gen_tests": lambda fn: [f"assert {fn}([-1, 2, 0, 3]) == [2, 3]", f"assert {fn}([-1, -2]) == []"]
    },
    {
        "category": "filters", "task_type": "sum_evens",
        "action": "the sum of only the even numbers in {a}.",
        "fn_names": ["sum_even_nums", "add_evens", "total_evens"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return sum(x for x in {a} if x % 2 == 0)",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2, 3, 4]) == 6", f"assert {fn}([1, 3, 5]) == 0", f"assert {fn}([]) == 0"]
    }
]

def get_args(arg_type):
    if arg_type == 1:
        return random.choice(P_1ARG_NUM)
    elif arg_type == "1_str":
        return random.choice(P_1ARG_STR)
    elif arg_type == "1_lst":
        return random.choice(P_1ARG_LST)
    elif arg_type == 2:
        return random.choice(P_2ARGS)
    elif arg_type == "2_lst":
        return random.choice(P_2ARG_LST)
    elif arg_type == "2_mix":
        return random.choice(P_2ARG_MIX)
    return ("a", "b")

def generate_example():
    task = random.choice(TASKS)
    fn_name = random.choice(task["fn_names"])
    args = get_args(task["arg_type"])
    
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIX_RETURN)
    
    if len(args) == 1:
        mid = random.choice(MID_1ARG).format(fn=fn_name, a=args[0])
        action = task["action"].format(a=args[0])
        code = task["code"].format(fn=fn_name, a=args[0])
    else:
        mid = random.choice(MID_2ARG).format(fn=fn_name, a=args[0], b=args[1])
        action = task["action"].format(a=args[0], b=args[1])
        code = task["code"].format(fn=fn_name, a=args[0], b=args[1])
        
    prompt = f"{prefix} {mid} {suffix} {action}"
    tests = task["gen_tests"](fn_name)
    
    return {
        "prompt": prompt,
        "target": code,
        "function_name": fn_name,
        "tests": tests,
        "task_type": task["task_type"],
        "category": task["category"]
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    eval_data = []
    seen_prompts = set()
    
    print("Generating adversarial holdout data...")
    attempts = 0
    while len(eval_data) < NUM_EVAL and attempts < 10000:
        attempts += 1
        ex = generate_example()
        
        if ex["prompt"] in seen_prompts:
            continue
        if not check_syntax(ex["target"]):
            continue
        if ex["function_name"] not in ex["target"]:
            continue
        if ex["function_name"] not in ex["prompt"]:
            continue
        if not check_execution(ex["target"], ex["tests"]):
            continue
        if "print" in ex["target"]:
            continue
            
        # Specific adversarial constraint: custom_abs cannot use `abs(`
        if ex["task_type"] == "custom_abs" and "abs(" in ex["target"]:
            continue
            
        seen_prompts.add(ex["prompt"])
        eval_data.append(ex)
        
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Generated {len(eval_data)} adversarial eval examples.")

if __name__ == "__main__":
    main()
