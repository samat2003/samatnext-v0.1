import os
import json
import random
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "stage2d_realistic_synthetic.jsonl")
EVAL_FILE = os.path.join(DATA_DIR, "stage2d_holdout_eval.jsonl")

NUM_TRAIN = 20000
NUM_EVAL = 200

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
# COMBINATORIAL TEMPLATES (To reach >20k unique without random numbers)
# ==============================================================================

PREFIXES = [
    "Write a Python function",
    "Create a Python function",
    "Implement the function",
    "Define a function",
    "Please write a function",
    "Write a method",
    "Construct a python function",
    "Could you write a function",
    "Create a python method",
    "Implement a routine",
    "Write code for a function",
    "Build a Python function",
    "Develop a function",
    "Program a Python function",
    "Draft a python function"
]

MID_1ARG = [
    "called {fn} that takes {a}",
    "named {fn} which accepts {a}",
    "{fn}({a})",
    "{fn} receiving {a}",
    "{fn} that processes {a}"
]

MID_2ARG = [
    "called {fn} that takes {a} and {b}",
    "named {fn} which accepts {a} and {b}",
    "{fn}({a}, {b})",
    "{fn} receiving {a} and {b}",
    "{fn} that processes {a} and {b}"
]

SUFFIX_RETURN = [
    "and returns",
    "to compute and return",
    "which calculates",
    "that outputs",
    "and outputs",
    "to return",
    "that evaluates to"
]

# Realistic argument name pools
P_2ARGS = [("a", "b"), ("x", "y"), ("num1", "num2"), ("val1", "val2"), ("left", "right"), ("first", "second"), ("n1", "n2"), ("v1", "v2"), ("p1", "p2"), ("arg1", "arg2")]
P_1ARG_NUM = [("n",), ("x",), ("num",), ("val",), ("value",), ("number",), ("p",), ("v",), ("count",), ("amount",)]
P_1ARG_STR = [("s",), ("string",), ("text",), ("msg",), ("word",), ("phrase",), ("line",), ("content",), ("chars",)]
P_1ARG_LST = [("lst",), ("items",), ("arr",), ("array",), ("elements",), ("nums",), ("values",), ("data",), ("sequence",), ("collection",)]
P_2ARG_LST = [("l1", "l2"), ("list1", "list2"), ("arr1", "arr2"), ("a", "b"), ("x", "y"), ("items1", "items2")]
P_2ARG_MIX = [("d", "key"), ("dict_obj", "k"), ("mapping", "key"), ("data", "k"), ("dictionary", "key"), ("hashmap", "k"), ("lookup", "key")]

TASKS = [
    # ---------------------------------------------------------
    # ARITHMETIC
    # ---------------------------------------------------------
    {
        "category": "arithmetic", "task_type": "add",
        "action": "their sum.",
        "fn_names": ["add", "sum_two", "add_numbers", "compute_sum", "get_sum", "addition", "add_vals", "calc_sum"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "gen_tests": lambda fn: [f"assert {fn}(2, 3) == 5", f"assert {fn}(-1, 1) == 0"]
    },
    {
        "category": "arithmetic", "task_type": "subtract",
        "action": "the difference of {a} minus {b}.",
        "fn_names": ["subtract", "sub", "difference", "minus", "sub_vals", "get_diff", "calc_diff", "subtract_nums"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return {a} - {b}",
        "gen_tests": lambda fn: [f"assert {fn}(5, 3) == 2", f"assert {fn}(1, 1) == 0"]
    },
    {
        "category": "arithmetic", "task_type": "multiply",
        "action": "their product.",
        "fn_names": ["multiply", "mul", "product", "times", "get_product", "calc_product", "multiply_nums"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return {a} * {b}",
        "gen_tests": lambda fn: [f"assert {fn}(5, 3) == 15", f"assert {fn}(0, 20) == 0"]
    },
    {
        "category": "arithmetic", "task_type": "divide",
        "action": "{a} divided by {b}.",
        "fn_names": ["divide", "div", "quotient", "calc_div", "divide_nums", "get_quotient"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return {a} / {b}",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 2", f"assert {fn}(1, 2) == 0.5"]
    },
    {
        "category": "arithmetic", "task_type": "abs",
        "action": "the absolute value of {a}.",
        "fn_names": ["absolute", "abs_val", "get_abs", "absolute_value", "calc_abs", "math_abs"],
        "arg_type": 1, "code": "def {fn}({a}):\n    return abs({a})",
        "gen_tests": lambda fn: [f"assert {fn}(6) == 6", f"assert {fn}(-10) == 10"]
    },
    {
        "category": "arithmetic", "task_type": "max",
        "action": "the maximum of {a} and {b}.",
        "fn_names": ["maximum", "get_max", "max_val", "find_max", "larger", "max_of_two"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return max({a}, {b})",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 6", f"assert {fn}(-1, 1) == 1"]
    },
    {
        "category": "arithmetic", "task_type": "min",
        "action": "the minimum of {a} and {b}.",
        "fn_names": ["minimum", "get_min", "min_val", "find_min", "smaller", "min_of_two"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return min({a}, {b})",
        "gen_tests": lambda fn: [f"assert {fn}(6, 3) == 3", f"assert {fn}(-1, 1) == -1"]
    },

    # ---------------------------------------------------------
    # STRINGS
    # ---------------------------------------------------------
    {
        "category": "strings", "task_type": "reverse",
        "action": "the reversed version of the string.",
        "fn_names": ["reverse_string", "rev_str", "reverse", "get_reversed", "string_reverse", "flip_string"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return {a}[::-1]",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 'olleh'", f"assert {fn}('') == ''"]
    },
    {
        "category": "strings", "task_type": "lowercase",
        "action": "the string converted to lowercase.",
        "fn_names": ["to_lowercase", "lower", "make_lower", "get_lower", "lowercase", "str_lower"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return {a}.lower()",
        "gen_tests": lambda fn: [f"assert {fn}('HeLLo') == 'hello'", f"assert {fn}('WORLD') == 'world'"]
    },
    {
        "category": "strings", "task_type": "uppercase",
        "action": "the string converted to uppercase.",
        "fn_names": ["to_uppercase", "upper", "make_upper", "get_upper", "uppercase", "str_upper"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return {a}.upper()",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 'HELLO'", f"assert {fn}('WorLd') == 'WORLD'"]
    },
    {
        "category": "strings", "task_type": "count_vowels",
        "action": "the number of vowels in the string.",
        "fn_names": ["count_vowels", "vowel_count", "num_vowels", "get_vowel_count", "find_vowels"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
        "gen_tests": lambda fn: [f"assert {fn}('hello') == 2", f"assert {fn}('xyz') == 0"]
    },
    {
        "category": "strings", "task_type": "contains_substring",
        "action": "True if {a} contains {b}, else False.",
        "fn_names": ["contains", "has_substring", "is_in", "string_contains", "check_substring"],
        "arg_type": 2, "code": "def {fn}({a}, {b}):\n    return {b} in {a}",
        "gen_tests": lambda fn: [f"assert {fn}('hello world', 'world') is True", f"assert {fn}('test', 'xyz') is False"]
    },

    # ---------------------------------------------------------
    # LISTS
    # ---------------------------------------------------------
    {
        "category": "lists", "task_type": "merge_lists",
        "action": "a new list that concatenates {a} and {b}.",
        "fn_names": ["merge", "merge_lists", "concat_lists", "join_lists", "combine", "append_lists"],
        "arg_type": "2_lst", "code": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2], [3, 4]) == [1, 2, 3, 4]", f"assert {fn}([], [1]) == [1]"]
    },
    {
        "category": "lists", "task_type": "first_element_safe",
        "action": "the first element of {a}, returning None if it is empty.",
        "fn_names": ["get_first_safe", "safe_first", "first_element", "get_first", "head_safe"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return {a}[0] if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2]) == 1", f"assert {fn}([]) is None"]
    },
    {
        "category": "lists", "task_type": "sum_list",
        "action": "the sum of all numbers in the list {a}.",
        "fn_names": ["sum_list", "list_sum", "get_sum", "calculate_sum", "total", "sum_elements"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return sum({a})",
        "gen_tests": lambda fn: [f"assert {fn}([1, 2, 3]) == 6", f"assert {fn}([]) == 0"]
    },
    {
        "category": "lists", "task_type": "max_list",
        "action": "the largest number in the list {a}.",
        "fn_names": ["max_list", "find_max", "get_max", "largest", "list_max", "maximum_item"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return max({a}) if {a} else None",
        "gen_tests": lambda fn: [f"assert {fn}([1, 5, 3]) == 5", f"assert {fn}([-1, -5]) == -1"]
    },
    {
        "category": "lists", "task_type": "remove_duplicates",
        "action": "a list with all duplicate elements removed.",
        "fn_names": ["remove_duplicates", "unique", "get_unique", "dedup", "make_unique", "distinct_items"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    return list(set({a}))",
        "gen_tests": lambda fn: [f"assert set({fn}([1, 2, 2, 3])) == {{1, 2, 3}}", f"assert {fn}([]) == []"]
    },

    # ---------------------------------------------------------
    # DICTS
    # ---------------------------------------------------------
    {
        "category": "dicts", "task_type": "get_value_safe",
        "action": "the value for {b} in dictionary {a}, returning None if not found.",
        "fn_names": ["get_safe", "safe_get", "get_value", "dict_get", "find_val", "lookup_key"],
        "arg_type": "2_mix", "code": "def {fn}({a}, {b}):\n    return {a}.get({b}, None)",
        "gen_tests": lambda fn: [f"assert {fn}({{'a': 1}}, 'a') == 1", f"assert {fn}({{}}, 'b') is None"]
    },
    {
        "category": "dicts", "task_type": "count_frequencies",
        "action": "a dictionary with the frequency count of each element in {a}.",
        "fn_names": ["count_freq", "frequencies", "get_counts", "count_elements", "item_counts", "tally"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    counts = {{}}\n    for x in {a}:\n        counts[x] = counts.get(x, 0) + 1\n    return counts",
        "gen_tests": lambda fn: [f"assert {fn}(['a', 'b', 'a']) == {{'a': 2, 'b': 1}}", f"assert {fn}([]) == {{}}"]
    },

    # ---------------------------------------------------------
    # BOOLEANS
    # ---------------------------------------------------------
    {
        "category": "booleans", "task_type": "is_even",
        "action": "True if {a} is even, and False otherwise.",
        "fn_names": ["is_even", "check_even", "even", "even_check", "is_number_even"],
        "arg_type": 1, "code": "def {fn}({a}):\n    return {a} % 2 == 0",
        "gen_tests": lambda fn: [f"assert {fn}(2) is True", f"assert {fn}(3) is False"]
    },
    {
        "category": "booleans", "task_type": "is_positive",
        "action": "True if {a} is strictly greater than zero, else False.",
        "fn_names": ["is_positive", "check_positive", "positive", "is_gt_zero", "is_val_positive"],
        "arg_type": 1, "code": "def {fn}({a}):\n    return {a} > 0",
        "gen_tests": lambda fn: [f"assert {fn}(5) is True", f"assert {fn}(-1) is False"]
    },
    {
        "category": "booleans", "task_type": "is_palindrome",
        "action": "True if the string {a} reads the same forwards and backwards.",
        "fn_names": ["is_palindrome", "check_palindrome", "palindrome", "is_pal", "is_str_palindrome"],
        "arg_type": "1_str", "code": "def {fn}({a}):\n    return {a} == {a}[::-1]",
        "gen_tests": lambda fn: [f"assert {fn}('racecar') is True", f"assert {fn}('hello') is False"]
    },

    # ---------------------------------------------------------
    # SIMPLE ALGORITHMS
    # ---------------------------------------------------------
    {
        "category": "simple_algorithms", "task_type": "factorial",
        "action": "the factorial of {a}.",
        "fn_names": ["factorial", "get_factorial", "calc_factorial", "fact", "compute_fact"],
        "arg_type": 1, "code": "def {fn}({a}):\n    if {a} == 0:\n        return 1\n    res = 1\n    for i in range(1, {a} + 1):\n        res *= i\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}(5) == 120", f"assert {fn}(0) == 1"]
    },
    {
        "category": "simple_algorithms", "task_type": "fibonacci",
        "action": "the {a}-th Fibonacci number (where fib(0)=0, fib(1)=1).",
        "fn_names": ["fibonacci", "fib", "get_fib", "calc_fib", "nth_fib", "fib_num"],
        "arg_type": 1, "code": "def {fn}({a}):\n    if {a} <= 0:\n        return 0\n    if {a} == 1:\n        return 1\n    a_, b_ = 0, 1\n    for _ in range(2, {a} + 1):\n        a_, b_ = b_, a_ + b_\n    return b_",
        "gen_tests": lambda fn: [f"assert {fn}(0) == 0", f"assert {fn}(5) == 5", f"assert {fn}(6) == 8"]
    },
    {
        "category": "simple_algorithms", "task_type": "flatten_one_level",
        "action": "a new list that is flattened by one level from {a}.",
        "fn_names": ["flatten", "flatten_list", "make_flat", "flat_list", "get_flattened", "flatten_arr"],
        "arg_type": "1_lst", "code": "def {fn}({a}):\n    res = []\n    for sublist in {a}:\n        res.extend(sublist)\n    return res",
        "gen_tests": lambda fn: [f"assert {fn}([[1, 2], [3]]) == [1, 2, 3]", f"assert {fn}([]) == []"]
    }
]

def get_args(arg_type):
    if arg_type == 1:
        return (random.choice(P_1ARG_NUM)[0],)
    elif arg_type == "1_str":
        return (random.choice(P_1ARG_STR)[0],)
    elif arg_type == "1_lst":
        return (random.choice(P_1ARG_LST)[0],)
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
    
    train_data = []
    eval_data = []
    seen_prompts = set()
    
    def generate_valid_example():
        # Because we have 25 tasks * ~150k combinations per task, 
        # it is extremely unlikely to hit an infinite loop trying to find 20k uniques.
        attempts = 0
        while attempts < 1000:
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
            if not ex["target"].strip():
                continue
                
            seen_prompts.add(ex["prompt"])
            return ex
        raise Exception("Failed to generate unique valid example after 1000 attempts. Combinatorial space might be exhausted.")
            
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
