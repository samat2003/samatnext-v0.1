import os, sys, json, ast, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval_suite import run_test_worker_subprocess

BAD_WORDS = ["# Test", "Test cases", "if __name__", "doctest", "unittest", "assert ", "print(", "input(", "```"]

def comprehensive_validate():
    data_dir = os.path.join(ROOT, "data")
    files = [
        "stage6A_blueprint_train.jsonl",
        "stage6A_blueprint_natural_holdout.jsonl",
        "stage6A_blueprint_holdout.jsonl" # hard holdout
    ]

    total_examples = 0
    total_failed = 0
    empty_count = 0
    
    random_samples = []
    
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
            
        print(f"\n[{fname}] Count: {len(data)}")
        total_examples += len(data)
        
        for ex in data:
            target = ex.get("target_completion", "")
            
            # Check for empty/comment-only
            if not target.strip() or target.strip().startswith("#") or target.strip().startswith('"""') or target.strip().startswith("'''"):
                empty_count += 1
                total_failed += 1
                print(f"Empty/Comment: {repr(target[:50])}")
                continue
                
            # Check for bad words
            for bw in BAD_WORDS:
                if bw in target:
                    total_failed += 1
                    print(f"Bad Word ({bw}): {repr(target[:50])}")
                    break
                    
            # Check indentation & top-level code
            lines = target.split("\n")
            bad_indent = False
            for line in lines:
                if line.strip() and not (line.startswith(" ") or line.startswith("\t")):
                    total_failed += 1
                    bad_indent = True
                    print(f"Bad Indent: {repr(line)}")
                    break
            if bad_indent: continue
                    
            # AST parse
            full_code = ex["prompt"] + "\n" + target
            try:
                ast.parse(full_code)
            except Exception as e:
                total_failed += 1
                print(f"AST Error: {e} | Code: {repr(full_code[:50])}")
                continue
                
            # Subprocess test execution
            # Extract function name from prompt naive way or use dict
            fn_name = ex.get("function_name", "func")
            tests_to_run = ex.get("tests", []) + ex.get("hidden_tests", [])
            # Timeout set to 2.0s
            test_pass, err_msg = run_test_worker_subprocess(full_code, fn_name, tests_to_run, 2.0)
            if not test_pass:
                total_failed += 1
                print(f"Test Failed: {err_msg} | Code: {repr(full_code[:50])}")
                
        random_samples.extend(random.sample(data, min(5, len(data))))
                
    print(f"\nTotal: {total_examples}, Failed: {total_failed}, Empty/Comments Only: {empty_count}")

if __name__ == "__main__":
    comprehensive_validate()
