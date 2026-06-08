import json
import itertools
import random
import os

TRAIN_VERBS = ["extract", "filter", "transform", "merge", "split", "calculate", "find", "group", "sort", "flatten", "reverse", "validate", "normalize", "map", "reduce", "aggregate", "convert", "parse", "format", "generate"]
TRAIN_NOUNS = ["strings", "integers", "floats", "booleans", "lists", "dictionaries", "tuples", "sets", "matrices", "graphs", "trees", "dates", "urls", "emails", "paths", "ip_addresses", "coordinates", "percentages", "currencies", "hex_codes"]
TRAIN_MODIFIERS = ["by_length", "by_value", "by_key", "conditionally", "recursively", "inplace", "safely", "with_defaults", "ignoring_errors", "strictly", "uniquely", "in_batches", "in_reverse", "case_insensitive", "to_strings", "to_ints", "nested"]

HOLDOUT_VERBS = ["scrub", "obfuscate", "tally", "partition", "collate", "weave", "interleave", "sift", "distill", "amalgamate", "reconcile", "bifurcate"]
HOLDOUT_NOUNS = ["tokens", "lexemes", "glyphs", "matrices", "tensors", "vertices", "edges", "centroids", "manifolds", "blobs", "chunks"]
HOLDOUT_MODIFIERS = ["heuristically", "stochastically", "deterministically", "iteratively", "symbolically", "topologically", "lexicographically", "chronologically", "spatially"]

DIFFICULTIES = ["easy", "medium", "hard"]
FORMATS = ["HumanEval-style function signature + docstring", "natural language function request", "examples-only prompt", "test-driven prompt", "bug-fix prompt", "partial-code completion prompt", "edge-case-heavy prompt"]

def generate_edge_cases(noun, modifier):
    cases = []
    if noun in ["strings", "lists", "tuples", "sets", "dictionaries"]:
        cases.append(f"empty {noun[:-1]}")
    if noun in ["integers", "floats"]:
        cases.append("zero")
        cases.append("negative values")
    if "nested" in modifier or noun in ["matrices", "trees", "graphs"]:
        cases.append("deeply nested structures")
    if "safely" in modifier or "ignoring_errors" in modifier:
        cases.append("None values")
        cases.append("wrong types")
    if not cases:
        cases.append("extremely large input size")
    return ", ".join(cases)

def generate_hidden_tests(fname, noun, modifier):
    tests = []
    if noun in ["strings", "lists", "tuples"]:
        tests.append(f"try:\n    res = {fname}([])\n    assert res is not None or res == []\nexcept:\n    pass")
    if noun == "strings":
        tests.append(f"try:\n    res = {fname}('')\n    assert res is not None or res == ''\nexcept:\n    pass")
    if noun == "dictionaries":
        tests.append(f"try:\n    res = {fname}({{}})\n    assert res is not None or res == {{}}\nexcept:\n    pass")
    if noun == "integers":
        tests.append(f"try:\n    res = {fname}(0)\nexcept:\n    pass")
        tests.append(f"try:\n    res = {fname}(-1)\nexcept:\n    pass")
    return tests

def build_blueprints(verbs, nouns, modifiers, prefix, is_holdout):
    combos = list(itertools.product(verbs, nouns, modifiers))
    random.seed(42 if is_holdout else 123)
    random.shuffle(combos)
    
    blueprints = []
    for i, (v, n, m) in enumerate(combos):
        fname = f"{prefix}{v}_{n}_{m}"
        edge_cases = generate_edge_cases(n, m)
        hidden_tests = generate_hidden_tests(fname, n, m)
        
        diff = random.choice(DIFFICULTIES)
        fmt = random.choice(FORMATS)
        
        args = ["data"]
        if "by_key" in m or "by_value" in m:
            args.append("key")
        if "safely" in m or "with_defaults" in m:
            args.append("default=None")
            
        bp = {
            "id": f"{prefix}{i}",
            "is_holdout": is_holdout,
            "task_family": n,
            "operation": v,
            "prompt_format": fmt,
            "difficulty": diff,
            "function_name": fname,
            "argument_names": args,
            "required_edge_cases": edge_cases,
            "hidden_tests": hidden_tests
        }
        blueprints.append(bp)
    return blueprints

def main():
    os.makedirs("data", exist_ok=True)
    
    train_bps = build_blueprints(TRAIN_VERBS, TRAIN_NOUNS, TRAIN_MODIFIERS, "", False)
    holdout_bps = build_blueprints(HOLDOUT_VERBS, HOLDOUT_NOUNS, HOLDOUT_MODIFIERS, "hld_", True)
    
    print(f"Generated {len(train_bps)} train blueprints and {len(holdout_bps)} holdout blueprints.")
    
    with open("data/stage6A_train_blueprints.jsonl", "w") as f:
        for bp in train_bps:
            f.write(json.dumps(bp) + "\n")
            
    with open("data/stage6A_holdout_blueprints.jsonl", "w") as f:
        for bp in holdout_bps:
            f.write(json.dumps(bp) + "\n")

if __name__ == "__main__":
    main()
