import os, sys, json, ast, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval_suite import run_test_worker_subprocess

def repair_datasets():
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
            
        print(f"\nProcessing {fname} ({len(data)} items)")
        
        repaired_data = []
        dropped = 0
        
        for ex in data:
            target = ex.get("target_completion", "")
            
            # 1. Fallback to extracting from target_code_full if target_completion is dead
            if not target.strip() or target.strip().startswith("#") or target.strip().startswith('"""'):
                full = ex.get("target_code_full", "")
                if full:
                    lines = full.split("\n")
                    body_lines = []
                    in_body = False
                    for line in lines:
                        if line.startswith("def "):
                            in_body = True
                            continue
                        if in_body and line.strip() and not (line.startswith(" ") or line.startswith("\t")):
                            break
                        if in_body:
                            body_lines.append(line)
                    target = "\n".join(body_lines) + "\n"
                    ex["target_completion"] = target

            # If it's STILL empty, drop it
            if not target.strip() or target.strip().startswith("#"):
                dropped += 1
                continue
                
            # Assume it's good enough if it has something indented
            bad_indent = False
            for line in target.split("\n"):
                if line.strip() and not (line.startswith(" ") or line.startswith("\t")):
                    bad_indent = True
                    break
            
            if bad_indent:
                dropped += 1
                continue
                
            # If valid, append
            repaired_data.append(ex)

        print(f"Dropped {dropped} unrepairable items. Remaining: {len(repaired_data)}")
        with open(fpath, "w", encoding="utf-8") as f:
            for ex in repaired_data:
                f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    repair_datasets()
