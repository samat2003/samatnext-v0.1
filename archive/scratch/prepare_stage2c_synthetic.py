import os
import json
import random
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "stage2c_synthetic_curriculum.jsonl")
EVAL_FILE = os.path.join(DATA_DIR, "stage2c_synthetic_eval.jsonl")

NUM_TRAIN = 10000
NUM_EVAL = 100

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
# TEMPLATES
# ==============================================================================

# Common wordings
W_2ARGS = [
    "Write a Python function {fn} that takes {a} and {b} and returns",
    "Create a Python function called {fn}({a}, {b}) to calculate",
    "Implement the function {fn} to compute",
    "Define a function {fn} that returns",
    "Please write a function {fn} that takes {a}, {b} and computes"
]

W_1ARG = [
    "Write a Python function {fn} that takes {a} and returns",
    "Create a Python function called {fn}({a}) to calculate",
    "Implement the function {fn} to compute",
    "Define a function {fn} that returns",
    "Please write a function {fn} that takes {a} and computes"
]

# Random parameter names
P_2ARGS = [("a", "b"), ("x", "y"), ("num1", "num2"), ("val1", "val2"), ("left", "right"), ("n1", "n2"), ("p1", "p2"), ("first", "second")]
P_1ARG_NUM = [("n",), ("x",), ("num",), ("val",), ("value",), ("number",), ("p",)]
P_1ARG_STR = [("s",), ("string",), ("text",), ("msg",), ("word",)]
P_1ARG_LST = [("lst",), ("items",), ("arr",), ("array",), ("elements",), ("nums",)]
P_2ARG_LST = [("l1", "l2"), ("list1", "list2"), ("arr1", "arr2"), ("a", "b")]
P_2ARG_MIX = [("d", "key"), ("dict_obj", "k"), ("mapping", "key"), ("data", "k")]

TASKS = [
    # ---------------------------------------------------------
    # ARITHMETIC
    # ---------------------------------------------------------
    {
        "category": "arithmetic", "task_type": "add",
        "wordings": [w + " their sum." for w in W_2ARGS],
        "fn_names": ["add", "sum_two", "add_numbers", "compute_sum", "get_sum", "addition", "add_vals", "sum_vals"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "gen_tests": lambda fn: [f"assert {fn}(2, 3) == 5", f"assert {fn}(-1, 1) == 0", f"assert {fn}(10, 20) == 30"]
    },
    {
        "category": "arithmetic", "task_type": "subtract",
        "wordings": [w + " the difference of {a} minus {b}." for w in W_2ARGS],
        "fn_names": ["subtract", "sub", "difference", "minus", "sub_vals", "get_diff"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return {a} - {b}",
        "gen_tests": lambda fn: [f"assert {fn}(5, 3) == 2", f"assert {fn}(1, 1) == 0", f"assert {fn}(10, 20) == -10"]
    },
    {
        "category": "arithmetic", "task_type": "multiply",
        "wordings": [w + " their product." for w in W_2ARGS],
        "fn_names": ["multiply", "mul", "product", "times", "get_product", "calc_product"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return {a} * {b}",
        "gen_tests": lambda fn: [f"assert {fn}(5, 3) == 15", f"assert {fn}(-1, 1) == -1", f"assert {fn}(0, 20) == 0"]
    },
    {
        "category": "arithmetic", "task_type": "divide",
        "wordings": [w + " {a} divided by {b}." for w in W_2ARGS],
        "fn_names": ["divide", "div", "quotient", "calc_div", "divide_nums"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return {a} / {b}",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 2", f"assert {fn}(10, 2) == 5", f"assert {fn}(1, 2) == 0.5"]
    },
    {
        "category": "arithmetic", "task_type": "abs",
        "wordings": [w + " the absolute value of {a}." for w in W_1ARG],
        "fn_names": ["absolute", "abs_val", "get_abs", "absolute_value", "calc_abs"],
        "args": P_1ARG_NUM,
        "code": "def {fn}({a}):\n    return abs({a})",
        "gen_tests": lambda fn: [f"assert {fn}(6) == 6", f"assert {fn}(-10) == 10", f"assert {fn}(0) == 0"]
    },
    {
        "category": "arithmetic", "task_type": "max",
        "wordings": [w + " the maximum of {a} and {b}." for w in W_2ARGS],
        "fn_names": ["maximum", "get_max", "max_val", "find_max", "larger"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return max({a}, {b})",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 6", f"assert {fn}(-1, 1) == 1", f"assert {fn}(20, 20) == 20"]
    },
    {
        "category": "arithmetic", "task_type": "min",
        "wordings": [w + " the minimum of {a} and {b}." for w in W_2ARGS],
        "fn_names": ["minimum", "get_min", "min_val", "find_min", "smaller"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return min({a}, {b})",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 3", f"assert {fn}(-1, 1) == -1", f"assert {fn}(20, 20) == 20"]
    },

    # ---------------------------------------------------------
    # STRINGS
    # ---------------------------------------------------------
    {
        "category": "strings", "task_type": "reverse",
        "wordings": [w + " the reversed version of the string." for w in W_1ARG],
        "fn_names": ["reverse_string", "rev_str", "reverse", "get_reversed", "string_reverse"],
        "args": P_1ARG_STR,
        "code": "def {fn}({a}):\n    return {a}[::-1]",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 'olleh'", f"assert {fn}('a') == 'a'", f"assert {fn}('') == ''"]
    },
    {
        "category": "strings", "task_type": "lowercase",
        "wordings": [w + " the string converted to lowercase." for w in W_1ARG],
        "fn_names": ["to_lowercase", "lower", "make_lower", "get_lower", "lowercase"],
        "args": P_1ARG_STR,
        "code": "def {fn}({a}):\n    return {a}.lower()",
        "gen_tests": lambda fn: [f"assert {fn}('HeLLo') == 'hello'", f"assert {fn}('WORLD') == 'world'"]
    },
    {
        "category": "strings", "task_type": "uppercase",
        "wordings": [w + " the string converted to uppercase." for w in W_1ARG],
        "fn_names": ["to_uppercase", "upper", "make_upper", "get_upper", "uppercase"],
        "args": P_1ARG_STR,
        "code": "def {fn}({a}):\n    return {a}.upper()",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 'HELLO'", f"assert {fn}('WorLd') == 'WORLD'"]
    },
    {
        "category": "strings", "task_type": "count_vowels",
        "wordings": [w + " the number of vowels in the string." for w in W_1ARG],
        "fn_names": ["count_vowels", "vowel_count", "num_vowels", "get_vowel_count"],
        "args": P_1ARG_STR,
        "code": "def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 2", f"assert {fn}('xyz') == 0", f"assert {fn}('AEIOU') == 5"]
    },
    {
        "category": "strings", "task_type": "contains_substring",
        "wordings": [w + " True if {a} contains {b}, else False." for w in W_2ARGS],
        "fn_names": ["contains", "has_substring", "is_in", "string_contains", "check_substring"],
        "args": P_2ARGS,
        "code": "def {fn}({a}, {b}):\n    return {b} in {a}",
        "gen_tests": lambda fn: [f"assert {fn}('hello world', 'world') is True", f"assert {fn}('test', 'xyz') is False"]
    },

    # ---------------------------------------------------------
    # LISTS
    # ---------------------------------------------------------
    {
        "category": "lists", "task_type": "merge_lists",
        "wordings": [w + " a new list that concatenates {a} and {b}." for w in W_2ARGS],
        "fn_names": ["merge", "merge_lists", "concat_lists", "join_lists", "combine"],
        "args": P_2ARG_LST,
        "code": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2], [3, 4]) == [1, 2, 3, 4]", f"assert {fn}([], [1]) == [1]"]
    },
    {
        "category": "lists", "task_type": "first_element_safe",
        "wordings": [w + " the first element of {a}, returning None if it is empty." for w in W_1ARG],
        "fn_names": ["get_first_safe", "safe_first", "first_element", "get_first", "head_safe"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    return {a}[0] if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2]) == 1", f"assert {fn}([]) is None"]
    },
    {
        "category": "lists", "task_type": "sum_list",
        "wordings": [w + " the sum of all numbers in the list {a}." for w in W_1ARG],
        "fn_names": ["sum_list", "list_sum", "get_sum", "calculate_sum", "total"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    return sum({a})",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2, 3]) == 6", f"assert {fn}([]) == 0"]
    },
    {
        "category": "lists", "task_type": "max_list",
        "wordings": [w + " the largest number in the list {a}." for w in W_1ARG],
        "fn_names": ["max_list", "find_max", "get_max", "largest", "list_max"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    return max({a}) if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([1, 5, 3]) == 5", f"assert {fn}([-1, -5]) == -1"]
    },
    {
        "category": "lists", "task_type": "remove_duplicates",
        "wordings": [w + " a list with all duplicate elements removed." for w in W_1ARG],
        "fn_names": ["remove_duplicates", "unique", "get_unique", "dedup", "make_unique"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    return list(set({a}))",
        "gen_tests": lambda fn: [f"assert set({fn}([1, 2, 2, 3])) == {{1, 2, 3}}", f"assert {fn}([]) == []"]
    },

    # ---------------------------------------------------------
    # DICTS
    # ---------------------------------------------------------
    {
        "category": "dicts", "task_type": "get_value_safe",
        "wordings": [w + " the value for {b} in dictionary {a}, returning None if not found." for w in W_2ARGS],
        "fn_names": ["get_safe", "safe_get", "get_value", "dict_get", "find_val"],
        "args": P_2ARG_MIX,
        "code": "def {fn}({a}, {b}):\n    return {a}.get({b}, None)",
        "gen_tests": lambda fn: [f"assert {fn}({{'a': 1}}, 'a') == 1", f"assert {fn}({{}}, 'b') is None"]
    },
    {
        "category": "dicts", "task_type": "count_frequencies",
        "wordings": [w + " a dictionary with the frequency count of each element in {a}." for w in W_1ARG],
        "fn_names": ["count_freq", "frequencies", "get_counts", "count_elements", "item_counts"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    counts = {{}}\n    for x in {a}:\n        counts[x] = counts.get(x, 0) + 1\n    return counts",
        "gen_tests": lambda fn: [f"assert {fn}(['a', 'b', 'a']) == {{'a': 2, 'b': 1}}", f"assert {fn}([]) == {{}}"]
    },

    # ---------------------------------------------------------
    # BOOLEANS
    # ---------------------------------------------------------
    {
        "category": "booleans", "task_type": "is_even",
        "wordings": [w + " True if {a} is even, and False otherwise." for w in W_1ARG],
        "fn_names": ["is_even", "check_even", "even", "even_check", "is_number_even"],
        "args": P_1ARG_NUM,
        "code": "def {fn}({a}):\n    return {a} % 2 == 0",
        "gen_tests": lambda fn: [f"assert {fn}(2) is True", f"assert {fn}(3) is False", f"assert {fn}(0) is True"]
    },
    {
        "category": "booleans", "task_type": "is_positive",
        "wordings": [w + " True if {a} is strictly greater than zero, else False." for w in W_1ARG],
        "fn_names": ["is_positive", "check_positive", "positive", "is_gt_zero"],
        "args": P_1ARG_NUM,
        "code": "def {fn}({a}):\n    return {a} > 0",
        "gen_tests": lambda fn: [f"assert {fn}(5) is True", f"assert {fn}(-1) is False", f"assert {fn}(0) is False"]
    },
    {
        "category": "booleans", "task_type": "is_palindrome",
        "wordings": [w + " True if the string {a} reads the same forwards and backwards." for w in W_1ARG],
        "fn_names": ["is_palindrome", "check_palindrome", "palindrome", "is_pal"],
        "args": P_1ARG_STR,
        "code": "def {fn}({a}):\n    return {a} == {a}[::-1]",
        "gen_tests": lambda fn: [f"assert {fn}('racecar') is True", f"assert {fn}('hello') is False", f"assert {fn}('a') is True"]
    },

    # ---------------------------------------------------------
    # SIMPLE ALGORITHMS
    # ---------------------------------------------------------
    {
        "category": "simple_algorithms", "task_type": "factorial",
        "wordings": [w + " the factorial of {a}." for w in W_1ARG],
        "fn_names": ["factorial", "get_factorial", "calc_factorial", "fact"],
        "args": P_1ARG_NUM,
        "code": "def {fn}({a}):\n    if {a} == 0:\n        return 1\n    res = 1\n    for i in range(1, {a} + 1):\n        res *= i\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}(5) == 120", f"assert {fn}(0) == 1", f"assert {fn}(3) == 6"]
    },
    {
        "category": "simple_algorithms", "task_type": "fibonacci",
        "wordings": [w + " the {a}-th Fibonacci number (where fib(0)=0, fib(1)=1)." for w in W_1ARG],
        "fn_names": ["fibonacci", "fib", "get_fib", "calc_fib", "nth_fib"],
        "args": P_1ARG_NUM,
        "code": "def {fn}({a}):\n    if {a} <= 0:\n        return 0\n    if {a} == 1:\n        return 1\n    a_, b_ = 0, 1\n    for _ in range(2, {a} + 1):\n        a_, b_ = b_, a_ + b_\n    return b_",
        "gen_tests": lambda fn: [f"assert {fn}(0) == 0", f"assert {fn}(1) == 1", f"assert {fn}(5) == 5", f"assert {fn}(6) == 8"]
    },
    {
        "category": "simple_algorithms", "task_type": "flatten_one_level",
        "wordings": [w + " a new list that is flattened by one level from {a}." for w in W_1ARG],
        "fn_names": ["flatten", "flatten_list", "make_flat", "flat_list", "get_flattened"],
        "args": P_1ARG_LST,
        "code": "def {fn}({a}):\n    res = []\n    for sublist in {a}:\n        res.extend(sublist)\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}([[1, 2], [3]]) == [1, 2, 3]", f"assert {fn}([]) == []"]
    }
]

def rand_arg(base_list):
    return random.choice(base_list) + str(random.randint(0, 999))
    
def get_args(task):
    a_type = task.get("arg_type", 1)
    if a_type == 1:
        return (rand_arg(["n", "x", "num", "val", "p", "v"]),)
    elif a_type == "1_str":
        return (rand_arg(["s", "string", "text", "msg", "word"]),)
    elif a_type == "1_lst":
        return (rand_arg(["lst", "arr", "items", "nums", "elements"]),)
    elif a_type == 2:
        return (rand_arg(["a", "x", "num1", "v1", "left"]), rand_arg(["b", "y", "num2", "v2", "right"]))
    elif a_type == "2_lst":
        return (rand_arg(["l1", "arr1", "list1", "a"]), rand_arg(["l2", "arr2", "list2", "b"]))
    elif a_type == "2_mix":
        return (rand_arg(["d", "data", "mapping"]), rand_arg(["k", "key", "idx"]))
    return ("a", "b")

def generate_example():
    task = random.choice(TASKS)
    wording = random.choice(task["wordings"])
    fn_name = random.choice(task["fn_names"])
    
    # We need to explicitly define arg_types inside TASKS or deduce it here based on category
    cat = task["category"]
    a_type = 1
    if cat == "strings" and task["task_type"] != "contains_substring":
        a_type = "1_str"
    elif cat == "strings" and task["task_type"] == "contains_substring":
        a_type = 2
    elif cat == "lists" and task["task_type"] == "merge_lists":
        a_type = "2_lst"
    elif cat == "lists":
        a_type = "1_lst"
    elif cat == "dicts" and task["task_type"] == "get_value_safe":
        a_type = "2_mix"
    elif cat == "dicts":
        a_type = "1_lst"
    elif cat in ["arithmetic"] and task["task_type"] not in ["abs", "factorial"]:
        a_type = 2
    elif cat == "arithmetic":
        a_type = 1
    
    task["arg_type"] = a_type
    args = get_args(task)
    
    if len(args) == 1:
        prompt = wording.format(fn=fn_name, a=args[0])
        code = task["code"].format(fn=fn_name, a=args[0])
    else:
        prompt = wording.format(fn=fn_name, a=args[0], b=args[1])
        code = task["code"].format(fn=fn_name, a=args[0], b=args[1])
        
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
    
    train_data = []
    eval_data = []
    
    seen_prompts = set()
    
    def generate_valid_example():
        while True:
            ex = generate_example()
            # 6. No duplicate prompts
            if ex["prompt"] in seen_prompts:
                continue
            # 1. target parses
            if not check_syntax(ex["target"]):
                continue
            # 2. function name exists in target
            if ex["function_name"] not in ex["target"]:
                continue
            # 3. function name matches prompt
            if ex["function_name"] not in ex["prompt"]:
                continue
            # 4. tests pass
            if not check_execution(ex["target"], ex["tests"]):
                continue
            # 5. no print-only (enforced by design: none of the templates use print)
            if "print" in ex["target"]:
                continue
            # 7. no empty targets
            if not ex["target"].strip():
                continue
                
            seen_prompts.add(ex["prompt"])
            return ex
            
    print("Generating training data...")
    for _ in range(NUM_TRAIN):
        train_data.append(generate_valid_example())
        
    print("Generating evaluation data...")
    for _ in range(NUM_EVAL):
        eval_data.append(generate_valid_example())
        
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
            
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Generated {len(train_data)} train and {len(eval_data)} eval examples.")

if __name__ == "__main__":
    main()
