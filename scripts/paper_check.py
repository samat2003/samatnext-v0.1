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
    # 4. arXiv claim hygiene and licensing checks
    print("\n--- 4. arXiv Claim Hygiene & Licensing Checks ---")
    hygiene_success = True
    try:
        readme_path = os.path.join(ROOT, "README.md")
        data_licenses_path = os.path.join(ROOT, "DATA_LICENSES.md")
        ckpt_path = os.path.join(ROOT, "CHECKPOINT_LICENSE.md")
        model_card_path = os.path.join(ROOT, "MODEL_CARD.md")
        table_md_path = os.path.join(ROOT, "results", "tables", "main_retention_table.md")
        table_json_path = os.path.join(ROOT, "results", "tables", "main_retention_table.json")
        
        readme = open(readme_path, "r", encoding="utf-8").read()
        data_licenses = open(data_licenses_path, "r", encoding="utf-8").read()
        ckpt = open(ckpt_path, "r", encoding="utf-8").read()
        model_card = open(model_card_path, "r", encoding="utf-8").read()
        table_md = open(table_md_path, "r", encoding="utf-8").read()
        with open(table_json_path, "r", encoding="utf-8") as f:
            table_json = json.load(f)
            
        # Checks on README
        stale_cmds = ["train_stage2a.py", "train_stage3.py", "train_stage5.py", "scripts/compare_models.py"]
        for cmd in stale_cmds:
            if cmd in readme:
                print(f"FAIL: README contains stale command: {cmd}")
                hygiene_success = False
                
        req_cmds = ["make test", "make reproduce-main-table", "--force-eval"]
        for cmd in req_cmds:
            if cmd not in readme:
                print(f"FAIL: README is missing required command/flag reference: {cmd}")
                hygiene_success = False
                
        fresh_vals = ["83.0%", "70.2%", "4.3%"]
        for val in fresh_vals:
            if val not in readme:
                print(f"FAIL: README is missing fresh result value: {val}")
                hygiene_success = False
                
        if "86.8%" in readme:
            print("FAIL: README contains stale result value: 86.8%")
            hygiene_success = False
            
        if "execution sandboxes" in readme.lower() or "subprocess sandboxing" in readme.lower():
            print("FAIL: README contains unsafe sandbox wording ('execution sandboxes' or 'subprocess sandboxing')")
            hygiene_success = False
            
        if "External artifact archive: pending" not in readme:
            print("FAIL: README must contain 'External artifact archive: pending' because no remote archive is set up yet.")
            hygiene_success = False
            
        if "CC BY-NC-SA 4.0" not in readme:
            print("FAIL: README is missing CC BY-NC-SA 4.0 reference.")
            hygiene_success = False
            
        # Checks on DATA_LICENSES.md
        if "Qwen Research license" not in data_licenses:
            print("FAIL: DATA_LICENSES.md is missing 'Qwen Research license' citation.")
            hygiene_success = False
            
        if "Qwen2.5-Coder-3B model is subject to the Qwen Research license" not in data_licenses:
            print("FAIL: DATA_LICENSES.md must explicitly cite the Qwen Research license for Qwen2.5-Coder-3B.")
            hygiene_success = False
            
        if "Qwen2.5-Coder-3B" in data_licenses and "Apache License 2.0 (under Qwen2.5-Coder model terms)" in data_licenses:
            print("FAIL: DATA_LICENSES.md claims Qwen2.5-Coder-3B is Apache-2.0.")
            hygiene_success = False
            
        if "Redistribution: Allowed. Committed under `data/stage5_teacher_distill.jsonl`" in data_licenses:
            print("FAIL: DATA_LICENSES.md claims Stage 5 SFT data redistribution is Allowed under data/stage5_teacher_distill.jsonl.")
            hygiene_success = False
            
        # Standardized warning in README.md
        std_warn = "Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a secure sandbox or complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or generated code."
        if std_warn not in readme:
            print("FAIL: README.md is missing the standardized subprocess-isolation warning.")
            hygiene_success = False
            
        # Checks on CHECKPOINT_LICENSE.md
        if "CC BY-NC-SA 4.0" not in ckpt:
            print("FAIL: CHECKPOINT_LICENSE.md is missing CC BY-NC-SA 4.0 statement.")
            hygiene_success = False
            
        # Checks on MODEL_CARD.md
        if "CC BY-NC-SA 4.0" not in model_card and "CHECKPOINT_LICENSE.md" not in model_card:
            print("FAIL: MODEL_CARD.md does not reference CC BY-NC-SA 4.0 or CHECKPOINT_LICENSE.md.")
            hygiene_success = False
            
        # Checks on tables
        for val in fresh_vals:
            if val not in table_md:
                print(f"FAIL: main_retention_table.md is missing fresh result value: {val}")
                hygiene_success = False
                
        row = table_json.get("SamatNext Curriculum lr=3e-6", {})
        if row.get("stage5_pass_rate") != 0.83:
            print(f"FAIL: JSON has incorrect stage5_pass_rate: {row.get('stage5_pass_rate')}")
            hygiene_success = False
        if row.get("stage3_retention_rate") != 0.702:
            print(f"FAIL: JSON has incorrect stage3_retention_rate: {row.get('stage3_retention_rate')}")
            hygiene_success = False
        if abs(row.get("stage2e_pass_rate", 0) - 0.043333333333333335) > 1e-12:
            print(f"FAIL: JSON has incorrect stage2e_pass_rate: {row.get('stage2e_pass_rate')}")
            hygiene_success = False
            
        # Checks on CITATION.cff
        citation_path = os.path.join(ROOT, "CITATION.cff")
        if os.path.exists(citation_path):
            try:
                import yaml
                with open(citation_path, "r", encoding="utf-8") as f:
                    yaml.safe_load(f.read())
                print("Checking CITATION.cff YAML parsing            ... PASS")
            except ImportError:
                # simpler syntax check fallback
                content = open(citation_path, "r", encoding="utf-8").read()
                if "cff-version:" in content and "title:" in content:
                    print("Checking CITATION.cff syntax fallback        ... PASS")
                else:
                    print("Checking CITATION.cff syntax fallback        ... FAIL")
                    hygiene_success = False
            except Exception as e:
                print(f"Checking CITATION.cff YAML parsing            ... FAIL ({e})")
                hygiene_success = False
        else:
            print("Checking CITATION.cff                         ... FAIL (missing)")
            hygiene_success = False
            
        if hygiene_success:
            print("PASS: All arXiv claim hygiene, licenses, and result-table synchronization checks passed.")
        else:
            success = False
    except Exception as e:
        print(f"FAIL: Error during arXiv claim hygiene checks: {e}")
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
