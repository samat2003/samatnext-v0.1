"""
Stage 4B: Explicit Identifier Copy Curriculum — Data Generator
==============================================================
Dataset:
  30,000 train examples
  2,000 eval examples

Composition:
  70% copy-only identity tasks (prompt repeats name 2-3 times)
  30% normal simple tasks

Name rules:
  snake_case, no digits, no syllable soup
  40% 2-token | 40% 3-token | 20% 4-token
  Eval names disjoint from train names
"""
import os, sys, ast, json, random
from itertools import product
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "stage4b_name_copy_train.jsonl")
EVAL_FILE  = os.path.join(DATA_DIR, "stage4b_name_copy_eval.jsonl")

NUM_TRAIN = 30_000
NUM_EVAL  =  2_000
SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# Word pools (readable, BPE-friendly, no digits, no syllable soup)
# ─────────────────────────────────────────────────────────────────────────────
VERBS = [
    "get", "find", "count", "check", "clean", "normalize", "reverse", "merge",
    "flatten", "filter", "remove", "collect", "convert", "compute", "calculate",
    "validate", "extract", "trim", "sort", "dedupe", "fetch", "lookup",
    "build", "make", "run", "apply", "process", "scan", "detect", "generate",
]
NOUNS = [
    "text", "string", "word", "words", "item", "items", "list", "values",
    "numbers", "scores", "data", "key", "value", "map", "vowels",
    "palindrome", "positives", "negatives", "elements", "pairs",
    "result", "output", "input", "entry", "records",
]
MODIFIERS = [
    "safe", "clean", "valid", "unique", "first", "last", "nested", "flat",
    "lower", "upper", "trimmed", "ordered", "even", "odd", "positive",
    "negative", "total", "minimum", "maximum", "all", "empty", "sorted",
]

def make_name_pool():
    """Generate readable 2-, 3-, 4-token names, categorized by length."""
    two   = set()
    three = set()
    four  = set()

    # 2-token: verb_noun, verb_modifier, modifier_noun
    for v, n in product(VERBS, NOUNS):
        two.add(f"{v}_{n}")
    for v, m in product(VERBS, MODIFIERS):
        two.add(f"{v}_{m}")
    for m, n in product(MODIFIERS, NOUNS):
        two.add(f"{m}_{n}")

    # 3-token: verb_modifier_noun, modifier_verb_noun, verb_noun_modifier
    for v, m, n in product(VERBS, MODIFIERS, NOUNS):
        nm = f"{v}_{m}_{n}"
        if len(nm) <= 36: three.add(nm)
    for m, v, n in product(MODIFIERS, VERBS, NOUNS):
        nm = f"{m}_{v}_{n}"
        if len(nm) <= 36: three.add(nm)
    for v, n, m in product(VERBS, NOUNS, MODIFIERS):
        nm = f"{v}_{n}_{m}"
        if len(nm) <= 36: three.add(nm)

    # 4-token: verb_modifier_verb_noun (subset)
    for v1, m, v2, n in product(VERBS[:16], MODIFIERS[:12], VERBS[:10], NOUNS[:14]):
        nm = f"{v1}_{m}_{v2}_{n}"
        if len(nm) <= 42: four.add(nm)
        if len(four) >= 40_000: break

    return list(two), list(three), list(four)

# ─────────────────────────────────────────────────────────────────────────────
# Argument pools
# ─────────────────────────────────────────────────────────────────────────────
ARGS_1 = [("x",), ("val",), ("data",), ("items",), ("text",), ("n",), ("value",),
          ("lst",), ("arr",), ("nums",), ("word",), ("string",), ("count",)]
ARGS_2 = [("a", "b"), ("x", "y"), ("val1", "val2"), ("left", "right"),
          ("first", "second"), ("n", "m"), ("num1", "num2")]

# ─────────────────────────────────────────────────────────────────────────────
# Copy-only prompt templates (name appears 2-3 times)
# ─────────────────────────────────────────────────────────────────────────────
COPY_TEMPLATES = [
    # User-specified variants
    "Write a Python function named {fn}. The function must be named exactly {fn}. It should return {a} unchanged.",
    "Create a function called {fn}. Use the exact name {fn}. Return the input {a}.",
    "Define {fn}({a}). Do not rename it. The function name must be {fn}. Return {a}.",
    "Implement a Python function {fn}. The def line must be exactly def {fn}({a}). Return the input.",
    # Additional strong-copy variants
    "Write a function. Its name must be exactly {fn}. Implement {fn}({a}) to return {a}.",
    "The function must be called {fn}. I repeat: name it {fn}. It receives {a} and returns {a}.",
    "Implement {fn}({a}). The function name is {fn} — do not change it. Return {a}.",
    "Create {fn}. The exact identifier is {fn}. It takes {a} and returns {a} as-is.",
    "Please implement a function with this exact name: {fn}. It accepts {a} and returns it unchanged.",
    "I need a function called exactly {fn}. Use {fn} as the function name. Return {a} from it.",
    "Your function must be def {fn}({a}). Do not substitute the name. Return {a}.",
    "Write Python code defining {fn}({a}). The identifier must be {fn}. It should return {a}.",
    "Define a function named {fn}. The name is {fn}. It returns {a} unchanged.",
    "Build a function. Function name: {fn}. Signature: {fn}({a}). Body: return {a}.",
    "Code a Python function {fn}({a}) that returns {a}. Name it {fn} exactly.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Normal task definitions (30%)
# ─────────────────────────────────────────────────────────────────────────────
def frames_1(action):
    return [
        f"Write a Python function {{fn}}({{a}}) that {action}.",
        f"Implement {{fn}}({{a}}) so that it {action}.",
        f"Create a function called {{fn}} that {action}, given {{a}}.",
        f"Define {{fn}} to accept {{a}} and {action}.",
        f"I need a function named {{fn}} that takes {{a}} and {action}.",
        f"Build a helper function {{fn}}({{a}}) that {action}.",
        f"Please implement {{fn}}({{a}}); it should {action}.",
        f"The function {{fn}}({{a}}) should {action}.",
        f"Provide a Python definition for {{fn}}({{a}}) that {action}.",
        f"Code up a function {{fn}}({{a}}) that {action}.",
    ]

def frames_2(action):
    return [
        f"Write a Python function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Implement {{fn}}({{a}}, {{b}}) so that it {action}.",
        f"Create a function called {{fn}} that {action}, taking {{a}} and {{b}} as parameters.",
        f"Define {{fn}} to accept {{a}} and {{b}} and {action}.",
        f"I need a function named {{fn}} that takes {{a}} and {{b}} and {action}.",
        f"Build a helper function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Please implement {{fn}}({{a}}, {{b}}); it should {action}.",
        f"The function {{fn}}({{a}}, {{b}}) should {action}.",
        f"Provide a Python definition for {{fn}}({{a}}, {{b}}) that {action}.",
        f"Code up a function {{fn}}({{a}}, {{b}}) that {action}.",
    ]

TASKS = [
    {"task": "add",            "arg_pool": ARGS_2, "arity": 2,
     "tmpl": frames_2("returns the sum of {a} and {b}"),
     "code": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} + {b}",
     "tests": lambda fn,a,b: [f"assert {fn}(2,3)==5", f"assert {fn}(-1,1)==0", f"assert {fn}(0,0)==0"]},
    {"task": "subtract",       "arg_pool": ARGS_2, "arity": 2,
     "tmpl": frames_2("returns {a} minus {b}"),
     "code": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} - {b}",
     "tests": lambda fn,a,b: [f"assert {fn}(5,3)==2", f"assert {fn}(1,1)==0"]},
    {"task": "multiply",       "arg_pool": ARGS_2, "arity": 2,
     "tmpl": frames_2("returns the product of {a} and {b}"),
     "code": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} * {b}",
     "tests": lambda fn,a,b: [f"assert {fn}(3,4)==12", f"assert {fn}(0,5)==0"]},
    {"task": "divide",         "arg_pool": ARGS_2, "arity": 2,
     "tmpl": frames_2("returns {a} divided by {b}"),
     "code": lambda fn,a,b: f"def {fn}({a}, {b}):\n    return {a} / {b}",
     "tests": lambda fn,a,b: [f"assert {fn}(10,2)==5.0", f"assert {fn}(9,3)==3.0"]},
    {"task": "absolute_value", "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns the absolute value of {a}"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return abs({a})",
     "tests": lambda fn,a,_=None: [f"assert {fn}(-5)==5", f"assert {fn}(3)==3", f"assert {fn}(0)==0"]},
    {"task": "is_even",        "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns True if {a} is even, False otherwise"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a} % 2 == 0",
     "tests": lambda fn,a,_=None: [f"assert {fn}(4)==True", f"assert {fn}(3)==False"]},
    {"task": "is_positive",    "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns True if {a} is strictly greater than zero, False otherwise"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a} > 0",
     "tests": lambda fn,a,_=None: [f"assert {fn}(5)==True", f"assert {fn}(-1)==False"]},
    {"task": "reverse_string", "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns {a} with its characters in reverse order"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}[::-1]",
     "tests": lambda fn,a,_=None: [f"assert {fn}('hello')=='olleh'", f"assert {fn}('')==''"]},
    {"task": "to_lowercase",   "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns {a} converted to all lowercase letters"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}.lower()",
     "tests": lambda fn,a,_=None: [f"assert {fn}('HELLO')=='hello'", f"assert {fn}('abc')=='abc'"]},
    {"task": "count_vowels",   "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("counts vowels (a, e, i, o, u) in {a} case-insensitively and returns the count"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
     "tests": lambda fn,a,_=None: [f"assert {fn}('hello')==2", f"assert {fn}('xyz')==0"]},
    {"task": "safe_first",     "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns the first element of {a}, or None if {a} is empty"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return {a}[0] if {a} else None",
     "tests": lambda fn,a,_=None: [f"assert {fn}([1,2,3])==1", f"assert {fn}([]) is None"]},
    {"task": "sum_list",       "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns the sum of all numbers in {a}"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return sum({a})",
     "tests": lambda fn,a,_=None: [f"assert {fn}([1,2,3])==6", f"assert {fn}([])==0"]},
    {"task": "filter_positives","arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns a list of only the positive (> 0) elements from {a}"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    return [x for x in {a} if x > 0]",
     "tests": lambda fn,a,_=None: [f"assert {fn}([-1,2,0,3])==[2,3]", f"assert {fn}([])==[]"]},
    {"task": "flatten",        "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("flattens {a} one level deep and returns a single list"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    result = []\n    for sub in {a}:\n        result.extend(sub)\n    return result",
     "tests": lambda fn,a,_=None: [f"assert {fn}([[1,2],[3]])==[1,2,3]", f"assert {fn}([])==[]"]},
    {"task": "unique_ordered", "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("removes duplicates from {a} while preserving insertion order"),
     "code": lambda fn,a,_=None: f"def {fn}({a}):\n    r=[]\n    for x in {a}:\n        if x not in r: r.append(x)\n    return r",
     "tests": lambda fn,a,_=None: [f"assert {fn}([1,2,1,3])==[1,2,3]", f"assert {fn}([])==[]"]},
    {"task": "factorial",      "arg_pool": ARGS_1, "arity": 1,
     "tmpl": frames_1("returns the factorial of {a}, where factorial(0) = 1"),
     "code": lambda fn,a,_=None: (
         f"def {fn}({a}):\n    if {a}==0: return 1\n"
         f"    r=1\n    for i in range(1,{a}+1): r*=i\n    return r"),
     "tests": lambda fn,a,_=None: [f"assert {fn}(0)==1", f"assert {fn}(5)==120"]},
]

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def ok_syntax(code):
    try: ast.parse(code); return True
    except: return False

def ok_exec(code, tests, fn_name):
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
        if fn_name not in ns: return False
        for t in tests:
            exec(compile(t, "<string>", "exec"), ns)
        return True
    except: return False

# ─────────────────────────────────────────────────────────────────────────────
# Record builder
# ─────────────────────────────────────────────────────────────────────────────
def make_record(uid, prompt, code, fn_name, args, tests, task_type, split):
    return {
        "id": f"stage4b_{split}_{uid:06d}",
        "prompt": prompt, "target": code, "function_name": fn_name,
        "arguments": list(args), "tests": tests,
        "task_type": task_type,
        "name_length": len(fn_name.split("_")),
        "difficulty": "trivial" if task_type == "copy_only" else "easy",
    }

# ─────────────────────────────────────────────────────────────────────────────
# Build copy-only examples
# ─────────────────────────────────────────────────────────────────────────────
def build_copy(names, split, n_wanted):
    records, skipped = [], 0
    rng_args = ARGS_1
    for fn_name in names:
        if len(records) >= n_wanted: break
        args = random.choice(rng_args)
        a = args[0]
        code  = f"def {fn_name}({a}):\n    return {a}"
        tests = [f"assert {fn_name}(42)==42", f"assert {fn_name}('x')=='x'"]
        if not ok_syntax(code) or not ok_exec(code, tests, fn_name):
            skipped += 1; continue
        tmpl  = random.choice(COPY_TEMPLATES)
        # Format: some templates use {a}, some don't
        try:
            prompt = tmpl.format(fn=fn_name, a=a)
        except KeyError:
            prompt = tmpl.format(fn=fn_name)
        records.append(make_record(len(records), prompt, code, fn_name, args,
                                   tests, "copy_only", split))
    if skipped: print(f"  [WARN] {skipped} copy examples skipped")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Build normal task examples
# ─────────────────────────────────────────────────────────────────────────────
def build_tasks(names, split, n_wanted):
    records, skipped = [], 0
    for fn_name in names:
        if len(records) >= n_wanted: break
        task = random.choice(TASKS)
        if task["arity"] == 1:
            args = random.choice(task["arg_pool"])
            a = args[0]
            code  = task["code"](fn_name, a)
            tests = task["tests"](fn_name, a)
            if not ok_syntax(code) or not ok_exec(code, tests, fn_name):
                skipped += 1; continue
            tmpl   = random.choice(task["tmpl"])
            prompt = tmpl.format(fn=fn_name, a=a)
        else:
            args = random.choice(task["arg_pool"])
            a, b = args
            code  = task["code"](fn_name, a, b)
            tests = task["tests"](fn_name, a, b)
            if not ok_syntax(code) or not ok_exec(code, tests, fn_name):
                skipped += 1; continue
            tmpl   = random.choice(task["tmpl"])
            prompt = tmpl.format(fn=fn_name, a=a, b=b)
        records.append(make_record(len(records), prompt, code, fn_name, list(args),
                                   tests, task["task"], split))
    if skipped: print(f"  [WARN] {skipped} task examples skipped")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("Stage 4B: Explicit Identifier Copy Curriculum — Generator")
    print("=" * 65)

    pool_2, pool_3, pool_4 = make_name_pool()
    random.shuffle(pool_2); random.shuffle(pool_3); random.shuffle(pool_4)
    print(f"Name pool — 2-token: {len(pool_2):,}  3-token: {len(pool_3):,}  4-token: {len(pool_4):,}")

    # Adaptive length distribution: use all 2-token, fill 3-token to balance
    # Target: ~40% 2-tok, ~40% 3-tok, ~20% 4-tok — but 2-tok pool is limited
    total_needed = NUM_TRAIN + NUM_EVAL + 500  # buffer
    n4 = int(total_needed * 0.20)
    n2 = min(len(pool_2), int(total_needed * 0.40))   # cap at pool size
    n3 = total_needed - n2 - n4                        # fill remainder with 3-token

    if len(pool_3) < n3:
        print(f"ERROR: 3-token pool ({len(pool_3):,}) < needed ({n3:,})")
        sys.exit(1)
    if len(pool_4) < n4:
        print(f"ERROR: 4-token pool ({len(pool_4):,}) < needed ({n4:,})")
        sys.exit(1)

    actual_pct_2 = n2 / total_needed * 100
    actual_pct_3 = n3 / total_needed * 100
    actual_pct_4 = n4 / total_needed * 100
    print(f"Name distribution  — 2-tok: {n2:,} ({actual_pct_2:.0f}%)  "
          f"3-tok: {n3:,} ({actual_pct_3:.0f}%)  "
          f"4-tok: {n4:,} ({actual_pct_4:.0f}%)")

    names_2 = pool_2[:n2]; names_3 = pool_3[:n3]; names_4 = pool_4[:n4]
    all_names = names_2 + names_3 + names_4
    random.shuffle(all_names)

    # Split train / eval (disjoint)
    eval_names  = all_names[:NUM_EVAL + 200]
    train_names = all_names[NUM_EVAL + 200:]

    assert len(set(eval_names) & set(train_names)) == 0, "Name overlap!"
    print(f"Train pool: {len(train_names):,}  |  Eval pool: {len(eval_names):,}  |  Overlap: 0")

    # Budget: 70% copy, 30% task
    COPY_TRAIN = int(NUM_TRAIN * 0.70)  # 21,000
    TASK_TRAIN = NUM_TRAIN - COPY_TRAIN # 9,000
    COPY_EVAL  = int(NUM_EVAL  * 0.70)  # 1,400
    TASK_EVAL  = NUM_EVAL  - COPY_EVAL  # 600

    # Build train
    train_copy_names = train_names[:COPY_TRAIN + 500]
    train_task_names = train_names[COPY_TRAIN + 500: COPY_TRAIN + 500 + TASK_TRAIN + 500]

    print(f"\nBuilding {COPY_TRAIN:,} copy-only (train)...")
    train_copy = build_copy(train_copy_names, "train", COPY_TRAIN)
    print(f"  Built: {len(train_copy):,}")

    print(f"Building {TASK_TRAIN:,} task examples (train)...")
    train_task = build_tasks(train_task_names, "train", TASK_TRAIN)
    print(f"  Built: {len(train_task):,}")

    # Build eval
    eval_copy_names = eval_names[:COPY_EVAL + 200]
    eval_task_names = eval_names[COPY_EVAL + 200:]

    print(f"Building {COPY_EVAL:,} copy-only (eval)...")
    eval_copy = build_copy(eval_copy_names, "eval", COPY_EVAL)
    print(f"  Built: {len(eval_copy):,}")

    print(f"Building {TASK_EVAL:,} task examples (eval)...")
    eval_task = build_tasks(eval_task_names, "eval", TASK_EVAL)
    print(f"  Built: {len(eval_task):,}")

    train_data = train_copy + train_task
    eval_data  = eval_copy  + eval_task
    random.shuffle(train_data); random.shuffle(eval_data)
    train_data = train_data[:NUM_TRAIN]
    eval_data  = eval_data[:NUM_EVAL]

    print(f"\n{'='*65}\nDIAGNOSTICS\n{'='*65}")
    print(f"Train: {len(train_data):,}  |  Eval: {len(eval_data):,}")

    cat = Counter(e["task_type"] for e in train_data)
    print("\nTrain task distribution:")
    for k,v in sorted(cat.items()): print(f"  {k:<30} {v:>6,}")

    nlen = Counter(e["name_length"] for e in train_data)
    print("\nTrain name-length distribution:")
    for k in sorted(nlen): print(f"  {k}-token: {nlen[k]:,}")

    print("\nFirst 8 training examples:")
    for i, ex in enumerate(train_data[:8], 1):
        print(f"\n[{i}] {ex['task_type']} | fn={ex['function_name']} ({ex['name_length']}-token)")
        print(f"    Prompt : {ex['prompt'][:100]}")
        print(f"    Code   : {ex['target'].splitlines()[0]}")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data: f.write(json.dumps(ex) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_data: f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(train_data):,} train -> {TRAIN_FILE}")
    print(f"Saved {len(eval_data):,} eval  -> {EVAL_FILE}")

if __name__ == "__main__":
    main()
