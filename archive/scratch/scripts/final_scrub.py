import os, sys, json, ast, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval_suite import run_test_worker_subprocess

BAD_WORDS = ["# Test", "Test cases", "if __name__", "doctest", "unittest", "assert ", "print(", "input(", "```"]

def strip_docstring(target):
    # If the target starts with a docstring, strip it.
    lines = target.split("\n")
    if not lines: return target
    
    in_docstring = False
    doc_char = None
    first_code_idx = 0
    
    # Check if first non-empty line is a docstring
    for i, line in enumerate(lines):
        sline = line.strip()
        if not sline:
            continue
        if sline.startswith('"""') or sline.startswith("'''"):
            in_docstring = True
            doc_char = sline[:3]
            # Check if it's a single-line docstring
            if len(sline) >= 6 and sline.endswith(doc_char):
                first_code_idx = i + 1
                in_docstring = False
            break
        else:
            # First non-empty line is not a docstring
            return target
            
    if in_docstring:
        for i in range(first_code_idx + 1, len(lines)):
            if doc_char in lines[i]:
                first_code_idx = i + 1
                break
                
    return "\n".join(lines[first_code_idx:])

def is_valid(ex):
    target = ex.get("target_completion", "")
    target = strip_docstring(target)
    
    if not target.strip() or target.strip().startswith("#"):
        return False, "Empty or comment only"
        
    for bw in BAD_WORDS:
        if bw in target:
            return False, f"Bad word: {bw}"
            
    bad_indent = False
    for line in target.split("\n"):
        if line.strip() and not (line.startswith(" ") or line.startswith("\t")):
            bad_indent = True
            break
    if bad_indent:
        return False, "Bad indentation"
        
    full_code = ex["prompt"] + "\n" + target
    try:
        ast.parse(full_code)
    except:
        return False, "AST parse failed"
        
    fn_name = ex.get("function_name", "func")
    tests_to_run = ex.get("tests", []) + ex.get("hidden_tests", [])
    
    if not tests_to_run:
        return False, "No tests"
        
    test_pass, err_msg = run_test_worker_subprocess(full_code, fn_name, tests_to_run, 2.0)
    if not test_pass:
        return False, f"Test failed: {err_msg}"
        
    ex["target_completion"] = target
    return True, ""

def final_scrub():
    data_dir = os.path.join(ROOT, "data")
    files = [
        "stage6A_blueprint_train.jsonl",
        "stage6A_blueprint_natural_holdout.jsonl",
        "stage6A_blueprint_holdout.jsonl"
    ]

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
            
        valid_data = []
        fail_reasons = {}
        
        for ex in data:
            ok, reason = is_valid(ex)
            if ok:
                valid_data.append(ex)
            else:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                
        print(f"\n[{fname}] Original: {len(data)}, Kept: {len(valid_data)}, Dropped: {len(data) - len(valid_data)}")
        for k, v in fail_reasons.items():
            print(f"  - {k}: {v}")
            
        with open(fpath, "w", encoding="utf-8") as f:
            for ex in valid_data:
                f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    final_scrub()
