"""
Stage 4 Readable-Name Copy Binding — Data Generator (v2: Amplified Copy Signal)
================================================================================
Rules:
  - All function names: snake_case, 2-4 meaningful English tokens
  - No digits, no syllable soup, no random gibberish
  - Eval names strictly disjoint from train names
  - Mix: 50% task examples + 50% explicit copy-only mini-curriculum
  - Copy-only templates use 'must be named exactly' phrasing
  - Task templates also include forced-name variants
  - All examples validated: ast.parse + sandboxed tests
"""
import os, sys, ast, json, random
from itertools import product
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "stage4_name_binding_train.jsonl")
EVAL_FILE  = os.path.join(DATA_DIR, "stage4_name_binding_eval.jsonl")

NUM_TRAIN  = 50_000
NUM_EVAL   = 1_000
SEED       = 42

# ──────────────────────────────────────────────────────────────────────────────
# Word pools (user-approved, BPE-friendly, readable)
# ──────────────────────────────────────────────────────────────────────────────
VERBS = [
    "get", "find", "count", "check", "clean", "normalize", "reverse", "merge",
    "flatten", "filter", "remove", "collect", "convert", "compute", "calculate",
    "validate", "extract", "trim", "sort", "dedupe", "safe", "fetch",
]

NOUNS = [
    "text", "string", "word", "words", "item", "items", "list", "lists",
    "values", "numbers", "scores", "data", "key", "value", "lookup", "map",
    "dictionary", "vowels", "prefix", "suffix", "substring", "palindrome",
    "positives", "negatives", "elements", "pairs",
]

MODIFIERS = [
    "safe", "clean", "valid", "unique", "first", "last", "nested", "flat",
    "lower", "upper", "trimmed", "ordered", "even", "odd", "positive",
    "negative", "all", "total", "minimum", "maximum",
]

def generate_all_names():
    """Generate combinatorial 2-, 3-, 4-token names, all unique."""
    names = set()
    # 2-token: verb_noun  (22x26=572)
    for v, n in product(VERBS, NOUNS):
        names.add(f"{v}_{n}")
    # 2-token: modifier_noun  (20x26=520)
    for m, n in product(MODIFIERS, NOUNS):
        names.add(f"{m}_{n}")
    # 2-token: verb_modifier  (22x20=440)
    for v, m in product(VERBS, MODIFIERS):
        names.add(f"{v}_{m}")
    # 3-token: verb_modifier_noun  (22x20x26=11,440)
    for v, m, n in product(VERBS, MODIFIERS, NOUNS):
        nm = f"{v}_{m}_{n}"
        if len(nm) <= 36: names.add(nm)
    # 3-token: modifier_verb_noun  (20x22x26=11,440)
    for m, v, n in product(MODIFIERS, VERBS, NOUNS):
        nm = f"{m}_{v}_{n}"
        if len(nm) <= 36: names.add(nm)
    # 3-token: verb_noun_modifier  (22x26x20=11,440)
    for v, n, m in product(VERBS, NOUNS, MODIFIERS):
        nm = f"{v}_{n}_{m}"
        if len(nm) <= 36: names.add(nm)
    # 4-token: verb_modifier_verb_noun (fill remainder)
    for v1, m, v2, n in product(VERBS[:14], MODIFIERS[:12], VERBS[:10], NOUNS[:14]):
        nm = f"{v1}_{m}_{v2}_{n}"
        if len(nm) <= 42: names.add(nm)
        if len(names) >= 80_000: break
    return list(names)

# ──────────────────────────────────────────────────────────────────────────────
# Argument pools
# ──────────────────────────────────────────────────────────────────────────────
POOL_2NUM = [("a","b"),("x","y"),("val1","val2"),("left","right"),("first","second"),("n","m")]
POOL_1NUM = [("n",),("x",),("value",),("number",),("count",),("val",)]
POOL_1STR = [("text",),("s",),("word",),("string",),("line",),("chars",)]
POOL_1LST = [("items",),("lst",),("arr",),("values",),("nums",),("data",),("elements",)]
POOL_2LST = [("a","b"),("list1","list2"),("left","right"),("arr1","arr2"),("data1","data2")]
POOL_2MIX = [("d","key"),("mapping","key"),("table","k"),("lookup","key")]

# ──────────────────────────────────────────────────────────────────────────────
# Paraphrase templates (20 per arity)
# ──────────────────────────────────────────────────────────────────────────────
def frames_2(action):
    return [
        f"Write a Python function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Create a function called {{fn}} that {action}, taking {{a}} and {{b}} as parameters.",
        f"Implement {{fn}}({{a}}, {{b}}) so that it {action}.",
        f"Define {{fn}} to accept {{a}} and {{b}} and {action}.",
        f"I need a Python function named {{fn}} that takes {{a}} and {{b}} and {action}.",
        f"Build a helper function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Given inputs {{a}} and {{b}}, write a function called {{fn}} that {action}.",
        f"Please implement {{fn}}({{a}}, {{b}}); it should {action}.",
        f"Could you write {{fn}}({{a}}, {{b}})? It should {action}.",
        f"Write code for a function {{fn}} that accepts {{a}} and {{b}} and {action}.",
        f"I want a function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Make a Python function named {{fn}} that accepts {{a}} and {{b}} and {action}.",
        f"Draft a function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Develop {{fn}}({{a}}, {{b}}) to {action}.",
        f"Program a Python function {{fn}}({{a}}, {{b}}) that {action}.",
        f"The function {{fn}}({{a}}, {{b}}) should {action}.",
        f"Provide a Python definition for {{fn}}({{a}}, {{b}}) that {action}.",
        f"Code up a function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Construct {{fn}}({{a}}, {{b}}) to {action}.",
        f"I require a Python function named {{fn}} that accepts {{a}} and {{b}} and {action}.",
    ]

def frames_1(action):
    return [
        f"Write a Python function {{fn}}({{a}}) that {action}.",
        f"Create a function called {{fn}} that {action}, given {{a}}.",
        f"Implement {{fn}}({{a}}) so that it {action}.",
        f"Define {{fn}} to accept {{a}} and {action}.",
        f"I need a Python function named {{fn}} that takes {{a}} and {action}.",
        f"Build a helper function {{fn}}({{a}}) that {action}.",
        f"Given input {{a}}, write a function called {{fn}} that {action}.",
        f"Please implement {{fn}}({{a}}); it should {action}.",
        f"Could you write {{fn}}({{a}})? It should {action}.",
        f"Write code for a function {{fn}} that accepts {{a}} and {action}.",
        f"I want a function {{fn}}({{a}}) that {action}.",
        f"Make a Python function named {{fn}} that accepts {{a}} and {action}.",
        f"Draft a function {{fn}}({{a}}) that {action}.",
        f"Develop {{fn}}({{a}}) to {action}.",
        f"Program a Python function {{fn}}({{a}}) that {action}.",
        f"The function {{fn}}({{a}}) should {action}.",
        f"Provide a Python definition for {{fn}}({{a}}) that {action}.",
        f"Code up a function {{fn}}({{a}}) that {action}.",
        f"Construct {{fn}}({{a}}) to {action}.",
        f"I require a Python function named {{fn}} that accepts {{a}} and {action}.",
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Task definitions (18 core tasks)
# ──────────────────────────────────────────────────────────────────────────────
TASKS = [
    {
        "task_type": "add", "category": "arithmetic", "arg_pool": POOL_2NUM,
        "templates": frames_2("returns the sum of {a} and {b}"),
        "code_fn": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} + {b}",
        "tests_fn": lambda fn,a,b: [f"assert {fn}(2,3)==5", f"assert {fn}(-1,1)==0", f"assert {fn}(0,0)==0"],
    },
    {
        "task_type": "subtract", "category": "arithmetic", "arg_pool": POOL_2NUM,
        "templates": frames_2("returns {a} minus {b}"),
        "code_fn": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} - {b}",
        "tests_fn": lambda fn,a,b: [f"assert {fn}(5,3)==2", f"assert {fn}(1,1)==0", f"assert {fn}(-1,-2)==1"],
    },
    {
        "task_type": "multiply", "category": "arithmetic", "arg_pool": POOL_2NUM,
        "templates": frames_2("returns the product of {a} and {b}"),
        "code_fn": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} * {b}",
        "tests_fn": lambda fn,a,b: [f"assert {fn}(3,4)==12", f"assert {fn}(0,5)==0", f"assert {fn}(-2,3)==-6"],
    },
    {
        "task_type": "divide", "category": "arithmetic", "arg_pool": POOL_2NUM,
        "templates": frames_2("returns {a} divided by {b}"),
        "code_fn": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} / {b}",
        "tests_fn": lambda fn,a,b: [f"assert {fn}(10,2)==5.0", f"assert {fn}(9,3)==3.0", f"assert {fn}(1,4)==0.25"],
    },
    {
        "task_type": "absolute_value", "category": "arithmetic", "arg_pool": POOL_1NUM,
        "templates": frames_1("returns the absolute value of {a}"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return abs({a})",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}(-5)==5", f"assert {fn}(3)==3", f"assert {fn}(0)==0"],
    },
    {
        "task_type": "is_even", "category": "booleans", "arg_pool": POOL_1NUM,
        "templates": frames_1("returns True if {a} is even, False otherwise"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a} % 2 == 0",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}(4)==True", f"assert {fn}(3)==False", f"assert {fn}(0)==True"],
    },
    {
        "task_type": "is_positive", "category": "booleans", "arg_pool": POOL_1NUM,
        "templates": frames_1("returns True if {a} is strictly greater than zero, False otherwise"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a} > 0",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}(5)==True", f"assert {fn}(-1)==False", f"assert {fn}(0)==False"],
    },
    {
        "task_type": "reverse_string", "category": "strings", "arg_pool": POOL_1STR,
        "templates": frames_1("returns {a} with its characters in reverse order"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}[::-1]",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}('hello')=='olleh'", f"assert {fn}('')==''", f"assert {fn}('a')=='a'"],
    },
    {
        "task_type": "to_lowercase", "category": "strings", "arg_pool": POOL_1STR,
        "templates": frames_1("returns {a} converted to all lowercase letters"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}.lower()",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}('HELLO')=='hello'", f"assert {fn}('abc')=='abc'"],
    },
    {
        "task_type": "to_uppercase", "category": "strings", "arg_pool": POOL_1STR,
        "templates": frames_1("returns {a} converted to all uppercase letters"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}.upper()",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}('hello')=='HELLO'", f"assert {fn}('ABC')=='ABC'"],
    },
    {
        "task_type": "count_vowels", "category": "strings", "arg_pool": POOL_1STR,
        "templates": frames_1("counts vowels (a, e, i, o, u) in {a} case-insensitively and returns the count"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}('hello')==2", f"assert {fn}('xyz')==0", f"assert {fn}('AEIOU')==5"],
    },
    {
        "task_type": "safe_first", "category": "lists", "arg_pool": POOL_1LST,
        "templates": frames_1("returns the first element of {a}, or None if {a} is empty"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}[0] if {a} else None",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}([1,2,3])==1", f"assert {fn}([]) is None", f"assert {fn}(['x'])=='x'"],
    },
    {
        "task_type": "sum_list", "category": "lists", "arg_pool": POOL_1LST,
        "templates": frames_1("returns the sum of all numbers in {a}"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return sum({a})",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}([1,2,3])==6", f"assert {fn}([])==0", f"assert {fn}([-1,1])==0"],
    },
    {
        "task_type": "filter_positives", "category": "lists", "arg_pool": POOL_1LST,
        "templates": frames_1("returns a list containing only the positive (greater than zero) elements from {a}"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    return [x for x in {a} if x > 0]",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}([-1,2,0,3])==[2,3]", f"assert {fn}([])==[]", f"assert {fn}([-5])==[]"],
    },
    {
        "task_type": "flatten", "category": "lists", "arg_pool": POOL_1LST,
        "templates": frames_1("flattens {a} by one level and returns a single list, handling empty sublists"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    result = []\n    for sub in {a}:\n        result.extend(sub)\n    return result",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}([[1,2],[3]])==[1,2,3]", f"assert {fn}([])==[]", f"assert {fn}([[],[1]])==[1]"],
    },
    {
        "task_type": "unique_preserve_order", "category": "lists", "arg_pool": POOL_1LST,
        "templates": frames_1("removes duplicates from {a} while preserving the original insertion order"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    result = []\n    for item in {a}:\n        if item not in result:\n            result.append(item)\n    return result",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}([1,2,1,3])==[1,2,3]", f"assert {fn}([])==[]", f"assert {fn}(['a','a'])==['a']"],
    },
    {
        "task_type": "safe_get", "category": "dicts", "arg_pool": POOL_2MIX,
        "templates": frames_2("returns the value for key {b} in dict {a}, or None if {b} is not present"),
        "code_fn": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a}.get({b}, None)",
        "tests_fn": lambda fn,a,b: [f"assert {fn}({{'x':1}},'x')==1", f"assert {fn}({{}},'k') is None", f"assert {fn}({{'a':'hi'}},'b') is None"],
    },
    {
        "task_type": "factorial", "category": "algorithms", "arg_pool": POOL_1NUM,
        "templates": frames_1("computes and returns the factorial of {a} (where factorial(0) = 1)"),
        "code_fn": lambda fn,a,_=None: f"def {fn}({a}):\n    if {a} == 0:\n        return 1\n    result = 1\n    for i in range(1, {a} + 1):\n        result *= i\n    return result",
        "tests_fn": lambda fn,a,_=None: [f"assert {fn}(0)==1", f"assert {fn}(5)==120", f"assert {fn}(3)==6"],
    },
]

# Copy-only task templates — explicit 'must be named exactly' phrasing
# The model must see the name in the prompt and output it verbatim in def
COPY_TEMPLATES_1ARG = [
    # Explicit 'named exactly' phrasing
    "Write a Python function. It must be named exactly {fn}. It takes {a} and returns {a} unchanged.",
    "Create a function. The function name must be exactly {fn}. It accepts {a} and returns {a}.",
    "Implement a function. Name it exactly {fn}. It should take {a} and return {a} as-is.",
    "Define a Python function with the exact name {fn}. It takes {a} and returns {a} directly.",
    "The function must be called exactly {fn}. It receives {a} and returns {a}.",
    "Write code for a function that must be named exactly {fn}. It returns {a} unchanged.",
    "Code up a function named {fn} — use that exact name. It accepts {a} and returns {a}.",
    # Inline signature phrasing (name appears in def-like position)
    "Write a Python function {fn}({a}) that returns {a} unchanged.",
    "Please implement {fn}({a}); it should return {a} as received.",
    "Build a helper function {fn}({a}) that simply returns its input {a}.",
    "The function {fn}({a}) should return {a} unchanged.",
    "Code up {fn}({a}) to return {a} without any changes.",
    "Implement {fn}({a}) that just returns {a} without modification.",
    "Define {fn}({a}) to return {a} directly.",
    # Repetition-based (name stated twice for copy-signal strength)
    "I need a function. It must be named {fn}. Implement {fn}({a}) to return {a}.",
    "Create {fn}. The function is called {fn} and it takes {a} and returns {a}.",
    "Write a function called {fn}. {fn} should accept {a} and return {a} as-is.",
    "Define a function named {fn}. The name is {fn}. It returns {a} unchanged.",
    "Provide a Python definition for {fn}({a}) that returns {a}.",
    "Draft a function {fn}({a}) that returns {a}.",
]
COPY_POOL_ARGS = [("x",),("val",),("data",),("items",),("text",),("n",),("value",)]

# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────
def valid_syntax(code):
    try: ast.parse(code); return True
    except: return False

def valid_exec(code, tests, fn_name):
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
        if fn_name not in ns: return False
        for t in tests:
            exec(compile(t, "<string>", "exec"), ns)
        return True
    except: return False

# ──────────────────────────────────────────────────────────────────────────────
# Build functions
# ──────────────────────────────────────────────────────────────────────────────
def make_record(uid, prompt, code, fn_name, args, tests, task_type, category, split, is_copy=False):
    toks = fn_name.split("_")
    return {
        "id": f"stage4_{split}_{uid:06d}",
        "prompt": prompt, "target": code, "function_name": fn_name,
        "arguments": list(args), "tests": tests,
        "task_type": "copy_only" if is_copy else task_type,
        "category": "copy_only" if is_copy else category,
        "name_length": len(toks),
        "name_style": "readable",
        "difficulty": "trivial" if is_copy else "easy",
    }

def build_task_examples(name_pool, split):
    """Build normal task examples. One unique name per example."""
    records = []
    skipped = 0
    name_iter = iter(name_pool)
    while len(records) < len(name_pool):
        try: fn_name = next(name_iter)
        except StopIteration: break
        task = random.choice(TASKS)
        args = random.choice(task["arg_pool"])
        if len(args) == 1:
            code  = task["code_fn"](fn_name, args[0])
            tests = task["tests_fn"](fn_name, args[0])
        else:
            code  = task["code_fn"](fn_name, args[0], args[1])
            tests = task["tests_fn"](fn_name, args[0], args[1])
        if not valid_syntax(code) or not valid_exec(code, tests, fn_name):
            skipped += 1; continue
        tmpl = random.choice(task["templates"])
        if len(args) == 1:
            prompt = tmpl.format(fn=fn_name, a=args[0])
        else:
            prompt = tmpl.format(fn=fn_name, a=args[0], b=args[1])
        records.append(make_record(len(records), prompt, code, fn_name, args,
                                   tests, task["task_type"], task["category"], split))
    if skipped:
        print(f"  [WARN] {skipped} task examples failed validation and were skipped.")
    return records

def build_copy_examples(name_pool, split):
    """Build copy-only examples. Teach name binding with trivial identity task."""
    records = []
    skipped = 0
    for fn_name in name_pool:
        args = random.choice(COPY_POOL_ARGS)
        a = args[0]
        code = f"def {fn_name}({a}):\n    return {a}"
        tests = [f"assert {fn_name}(42)==42", f"assert {fn_name}('x')=='x'"]
        if not valid_syntax(code) or not valid_exec(code, tests, fn_name):
            skipped += 1; continue
        tmpl = random.choice(COPY_TEMPLATES_1ARG)
        prompt = tmpl.format(fn=fn_name, a=a)
        records.append(make_record(len(records), prompt, code, fn_name, args,
                                   tests, "copy_only", "copy_only", split, is_copy=True))
    if skipped:
        print(f"  [WARN] {skipped} copy examples failed validation and were skipped.")
    return records

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("Stage 4 Readable-Name Copy Binding — Dataset Generator")
    print("=" * 65)

    # Generate all candidate names
    all_names = generate_all_names()
    random.shuffle(all_names)
    print(f"Total unique readable names generated: {len(all_names):,}")

    # Budget: 50% task, 50% copy-only (amplified copy signal)
    TASK_TRAIN = int(NUM_TRAIN * 0.50)   # 25,000 task examples
    COPY_TRAIN = NUM_TRAIN - TASK_TRAIN  # 25,000 copy-only examples

    # Eval: also 50/50
    TASK_EVAL  = int(NUM_EVAL * 0.50)    # 500
    COPY_EVAL  = NUM_EVAL - TASK_EVAL    # 500

    total_needed = TASK_TRAIN + COPY_TRAIN + TASK_EVAL + COPY_EVAL
    if len(all_names) < total_needed:
        print(f"ERROR: Only {len(all_names):,} unique names, need {total_needed:,}. Expand pools.")
        sys.exit(1)

    # Assign disjoint name pools
    idx = 0
    train_task_names = all_names[idx : idx + TASK_TRAIN];  idx += TASK_TRAIN
    train_copy_names = all_names[idx : idx + COPY_TRAIN];  idx += COPY_TRAIN
    eval_task_names  = all_names[idx : idx + TASK_EVAL];   idx += TASK_EVAL
    eval_copy_names  = all_names[idx : idx + COPY_EVAL];   idx += COPY_EVAL

    # Verify disjoint
    train_all_names = set(train_task_names) | set(train_copy_names)
    eval_all_names  = set(eval_task_names)  | set(eval_copy_names)
    overlap = train_all_names & eval_all_names
    assert len(overlap) == 0, f"FATAL: {len(overlap)} name(s) overlap between train and eval!"
    print(f"Train names : {len(train_all_names):,}  |  Eval names: {len(eval_all_names):,}  |  Overlap: 0")

    # Build examples
    print(f"\nBuilding {TASK_TRAIN:,} task examples (train)...")
    train_task = build_task_examples(train_task_names, "train")
    print(f"  Built: {len(train_task):,}")

    print(f"Building {COPY_TRAIN:,} copy-only examples (train)...")
    train_copy = build_copy_examples(train_copy_names, "train")
    print(f"  Built: {len(train_copy):,}")

    print(f"Building {TASK_EVAL:,} task examples (eval)...")
    eval_task  = build_task_examples(eval_task_names, "eval")
    print(f"  Built: {len(eval_task):,}")

    print(f"Building {COPY_EVAL:,} copy-only examples (eval)...")
    eval_copy  = build_copy_examples(eval_copy_names, "eval")
    print(f"  Built: {len(eval_copy):,}")

    # Mix and shuffle
    train_data = train_task + train_copy
    eval_data  = eval_task  + eval_copy
    random.shuffle(train_data)
    random.shuffle(eval_data)

    # Trim to exact sizes (in case of validation skips)
    train_data = train_data[:NUM_TRAIN]
    eval_data  = eval_data[:NUM_EVAL]

    # Diagnostics
    print(f"\n{'='*65}\nDIAGNOSTICS\n{'='*65}")
    print(f"Train size  : {len(train_data):,}")
    print(f"Eval size   : {len(eval_data):,}")

    cat_dist = Counter(e["category"] for e in train_data)
    print("\nTrain category distribution:")
    for k, v in sorted(cat_dist.items()):
        print(f"  {k:<30} {v:>6,}")

    len_dist = Counter(e["name_length"] for e in train_data)
    print("\nTrain name_length distribution:")
    for k in sorted(len_dist):
        print(f"  {k}-token: {len_dist[k]:,}")

    print("\nFirst 10 training examples:")
    for i, ex in enumerate(train_data[:10], 1):
        first_line = ex["target"].splitlines()[0]
        print(f"\n[{i}] task={ex['task_type']} | fn={ex['function_name']}")
        print(f"    Prompt : {ex['prompt'][:90]}")
        print(f"    Code   : {first_line}")

    print("\nFirst 5 eval examples:")
    for i, ex in enumerate(eval_data[:5], 1):
        print(f"\n[{i}] task={ex['task_type']} | fn={ex['function_name']}")
        print(f"    Prompt : {ex['prompt'][:90]}")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data: f.write(json.dumps(ex) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_data: f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(train_data):,} train  -> {TRAIN_FILE}")
    print(f"Saved {len(eval_data):,} eval   -> {EVAL_FILE}")

if __name__ == "__main__":
    main()
