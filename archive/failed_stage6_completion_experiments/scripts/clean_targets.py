import os
import json

def clean_completion(text):
    lines = text.split("\n")
    clean_lines = []
    
    # We want to keep all indented lines (or empty lines between indented blocks).
    # Once we hit a line that is NOT empty and has NO leading spaces/tabs (indentation == 0),
    # we assume that's the end of the intended body (e.g. '# Test cases', 'if __name__ == ...').
    for line in lines:
        if line.strip() == "":
            clean_lines.append(line)
        elif line.startswith(" ") or line.startswith("\t"):
            clean_lines.append(line)
        else:
            # We hit an unindented line with text. This is outside the function body.
            break
            
    # Remove trailing empty lines
    while clean_lines and clean_lines[-1].strip() == "":
        clean_lines.pop()
        
    return "\n".join(clean_lines) + "\n"

def process_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
        
    modified = 0
    for ex in data:
        orig = ex.get("target_completion", "")
        if orig:
            cleaned = clean_completion(orig)
            if cleaned != orig:
                ex["target_completion"] = cleaned
                modified += 1
                
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
            
    print(f"  -> Modified {modified} out of {len(data)} records.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    files = [
        "stage6A_blueprint_train.jsonl",
        "stage6A_blueprint_natural_holdout.jsonl",
        "stage6A_blueprint_hard_holdout.jsonl"
    ]
    
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            process_file(fpath)
