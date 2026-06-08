import os, sys, json, ast, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval_suite import run_test_worker_subprocess

BAD_WORDS = ["# Test", "Test cases", "if __name__", "doctest", "unittest", "assert ", "print(", "input(", "```"]

def validate_datasets():
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
            if not target.strip() or target.strip().startswith("#"):
                empty_count += 1
                continue
                
            for bw in BAD_WORDS:
                if bw in target:
                    total_failed += 1
                    break
                    
            lines = target.split("\n")
            for line in lines:
                if line.strip() and not (line.startswith(" ") or line.startswith("\t")):
                    total_failed += 1
                    break
                    
            full_code = ex["prompt"] + "\n" + target
            try:
                ast.parse(full_code)
            except:
                total_failed += 1
                
        random_samples.extend(random.sample(data, min(5, len(data))))
                
    print(f"\nTotal: {total_examples}, Failed: {total_failed}, Empty/Comments Only: {empty_count}")
    
    print("\n--- Random Cleaned Targets ---")
    for s in random_samples[:3]:
        print(repr(s.get("target_completion")))
        
if __name__ == "__main__":
    validate_datasets()
