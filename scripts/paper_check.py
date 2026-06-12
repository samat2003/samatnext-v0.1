# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

def check_files_exist():
    required_files = [
        "README.md",
        "DATA_LICENSES.md",
        "CHECKPOINT_LICENSE.md",
        "MODEL_CARD.md",
        "results/tables/main_retention_table.md",
        "results/tables/main_retention_table.json",
    ]
    for rel_path in required_files:
        if not (ROOT / rel_path).exists():
            print(f"FAIL: Required file missing: {rel_path}")
            return False
    return True

def check_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    
    stale_cmds = ["train_stage2a.py", "train_stage3.py", "train_stage5.py", "scripts/compare_models.py"]
    for cmd in stale_cmds:
        if cmd in readme:
            print(f"FAIL: README has stale command {cmd}")
            return False
            
    if "83.0%" not in readme or "70.2%" not in readme or "4.3%" not in readme:
        print("FAIL: README missing fresh result values")
        return False
        
    if "86.8%" in readme:
        print("FAIL: README has stale result value 86.8%")
        return False
        
    lower_readme = readme.lower()
    if "execution sandboxes" in lower_readme or "subprocess sandboxing" in lower_readme:
        print("FAIL: README has unsafe sandbox wording")
        return False
        
    if "fresh_eval_<timestamp>" not in readme:
        print("FAIL: README missing fresh_eval_<timestamp>")
        return False
        
    if "fresh_eval_<timestamp>/" in readme:
        print("FAIL: README contains fresh_eval_<timestamp>/")
        return False
        
    if "CC BY-NC-SA 4.0" not in readme:
        print("FAIL: README missing CC BY-NC-SA 4.0")
        return False
        
    return True

def check_licenses():
    data_licenses = (ROOT / "DATA_LICENSES.md").read_text(encoding="utf-8")
    if "Qwen Research license" not in data_licenses:
        print("FAIL: DATA_LICENSES.md missing Qwen Research license")
        return False
        
    if "Apache-2.0" in data_licenses and "Qwen2.5-Coder-3B model is subject to the Qwen Research license, not Apache-2.0" not in data_licenses:
        if "Qwen2.5-Coder-3B outputs are Apache-2.0" in data_licenses or "Qwen2.5-Coder-3B is Apache-2.0" in data_licenses:
            print("FAIL: DATA_LICENSES.md claims Qwen2.5-Coder-3B is Apache-2.0")
            return False
            
    ckpt = (ROOT / "CHECKPOINT_LICENSE.md").read_text(encoding="utf-8")
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    
    if "CC BY-NC-SA 4.0" not in ckpt:
        print("FAIL: CHECKPOINT_LICENSE.md missing CC BY-NC-SA 4.0")
        return False
        
    if "CC BY-NC-SA 4.0" not in model_card and "CHECKPOINT_LICENSE.md" not in model_card:
        print("FAIL: MODEL_CARD.md missing CC BY-NC-SA 4.0 or CHECKPOINT_LICENSE.md reference")
        return False
        
    return True

def check_results():
    table_md = (ROOT / "results" / "tables" / "main_retention_table.md").read_text(encoding="utf-8")
    if "83.0%" not in table_md or "70.2%" not in table_md or "4.3%" not in table_md:
        print("FAIL: main_retention_table.md missing fresh result values")
        return False
        
    table_json_path = ROOT / "results" / "tables" / "main_retention_table.json"
    if table_json_path.exists():
        data = json.loads(table_json_path.read_text(encoding="utf-8"))
        row = data.get("SamatNext Curriculum lr=3e-6", {})
        if row.get("stage5_pass_rate") != 0.83:
            print("FAIL: JSON has incorrect stage5_pass_rate")
            return False
        if row.get("stage3_retention_rate") != 0.702:
            print("FAIL: JSON has incorrect stage3_retention_rate")
            return False
            
    return True

def main():
    print("Running paper checks...")
    success = check_files_exist() and check_readme() and check_licenses() and check_results()
    if success:
        print("PASS: All checks passed")
        sys.exit(0)
    else:
        print("FAIL: Some checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
