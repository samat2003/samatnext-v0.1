# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_file_exists(rel_path):
    path = os.path.join(ROOT, rel_path)
    exists = os.path.exists(path)
    print(f"Checking {rel_path:<40} ... {'PASS' if exists else 'FAIL'}")
    return exists

def main():
    print("=== RUNNING SAMATNEXT-V0.1 PAPER & ARXIV-READY CHECKLIST ===")
    
    success = True
    
    # 1. License separation files
    license_files = [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "DATA_LICENSES.md",
        "MODEL_CARD.md",
        "CHECKPOINT_LICENSE.md",
        "SECURITY.md",
        "CITATION.cff"
    ]
    print("\n--- 1. Licensing & Metadata Separations ---")
    for lf in license_files:
        if not check_file_exists(lf):
            success = False
            
    # 2. Main Tables & JSON outputs
    result_files = [
        "results/tables/main_retention_table.json",
        "results/tables/main_retention_table.md",
        "results/tables/parameter_counts.json",
        "results/tables/parameter_counts.md",
        "results/tables/vram_benchmark.json",
        "reports/vram_benchmark.md",
        "reports/contamination_report.md"
    ]
    print("\n--- 2. Generated JSON / MD Scientific Artifacts ---")
    for rf in result_files:
        if not check_file_exists(rf):
            success = False
            
    # 3. Model Parameter Matching Validation
    print("\n--- 3. Parameter Counts Truth Check ---")
    param_path = os.path.join(ROOT, "results", "tables", "parameter_counts.json")
    if os.path.exists(param_path):
        try:
            with open(param_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            samat_params = data.get("samatnext", {}).get("total_parameters", 0)
            trans_params = data.get("transformer", {}).get("total_parameters", 0)
            diff = abs(samat_params - trans_params)
            pct_diff = (diff / max(samat_params, trans_params, 1)) * 100
            
            print(f"SamatNext parameters  : {samat_params:,}")
            print(f"Transformer parameters: {trans_params:,}")
            print(f"Difference            : {diff:,} ({pct_diff:.6f}%)")
            
            if pct_diff > 0.01:
                print("FAIL: Parameter counts differ by more than 0.01%!")
                success = False
            else:
                print("PASS: Parameter counts match (differ by <0.01%).")
        except Exception as e:
            print(f"FAIL: Error parsing parameter counts: {e}")
            success = False
    else:
        print("FAIL: parameter_counts.json is missing.")
        success = False
        
    print("\n==================================================")
    if success:
        print("=== ALL PAPER CHECKLIST VALIDATION GATES PASSED ===")
        print("==================================================")
        sys.exit(0)
    else:
        print("=== VALIDATION GATES FAILED! Please generate missing results. ===")
        print("==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
