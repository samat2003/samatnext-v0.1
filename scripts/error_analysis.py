# SPDX-License-Identifier: Apache-2.0
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "SamatNext Curriculum": {
        "stage5": os.path.join(ROOT, "results", "eval_stage_5_teacher-style_holdout.json"),
        "stage3": os.path.join(ROOT, "results", "eval_stage_3_paraphrase.json"),
        "stage2e": os.path.join(ROOT, "results", "eval_stage_2e_adversarial.json")
    },
    "Transformer Curriculum": {
        "stage5": os.path.join(ROOT, "results", "eval_transformer_curriculum_stage_5_teacher-style_holdout.json"),
        "stage3": os.path.join(ROOT, "results", "eval_transformer_curriculum_stage_3_paraphrase.json"),
        "stage2e": os.path.join(ROOT, "results", "eval_transformer_curriculum_stage_2e_adversarial.json")
    }
}

def analyze_file(filepath):
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total = len(data)
    passed = 0
    syntax_errors = 0
    assertion_errors = 0
    name_errors = 0
    timeouts = 0
    other_errors = 0
    
    for item in data:
        if item.get("test_pass", False):
            passed += 1
            continue
            
        err = item.get("err_msg", "").strip()
        syntax_ok = item.get("syntax_ok", True)
        
        if not syntax_ok or "SyntaxError" in err:
            syntax_errors += 1
        elif "AssertionError" in err:
            assertion_errors += 1
        elif "NameError" in err or "AttributeError" in err:
            name_errors += 1
        elif "Timeout" in err or "timeout" in err.lower():
            timeouts += 1
        else:
            other_errors += 1
            
    return {
        "total": total,
        "passed": passed,
        "syntax_errors": syntax_errors,
        "assertion_errors": assertion_errors,
        "name_errors": name_errors,
        "timeouts": timeouts,
        "other_errors": other_errors
    }

def main():
    report_data = {}
    
    for model_name, stages in MODELS.items():
        report_data[model_name] = {}
        for stage_name, filepath in stages.items():
            stats = analyze_file(filepath)
            if stats:
                report_data[model_name][stage_name] = stats
                
    report_path = os.path.join(ROOT, "reports", "error_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Scientific Error Analysis Report\n\n")
        f.write("This report categorizes the failure modes of the curriculum-trained **SamatNext-v0.1** and the matched **Transformer** baseline across the evaluation stages. Errors are categorized using subprocess isolation output.\n\n")
        
        for model_name, stages in report_data.items():
            f.write(f"## {model_name}\n\n")
            
            for stage_name, stats in stages.items():
                pass_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
                f.write(f"### {stage_name.upper()} (Total Tasks: {stats['total']})\n")
                f.write(f"- **Pass Rate:** {pass_rate:.1f}%\n")
                f.write("| Failure Mode | Count | Percentage of Failures |\n")
                f.write("| :--- | :---: | :---: |\n")
                
                failures = stats["total"] - stats["passed"]
                if failures == 0:
                    f.write("| *No failures recorded* | - | - |\n\n")
                    continue
                    
                def format_row(name, count):
                    pct = (count / failures) * 100
                    return f"| {name} | {count} | {pct:.1f}% |\n"
                    
                f.write(format_row("SyntaxError / IndentationError", stats["syntax_errors"]))
                f.write(format_row("AssertionError", stats["assertion_errors"]))
                f.write(format_row("NameError / AttributeError", stats["name_errors"]))
                f.write(format_row("Timeout", stats["timeouts"]))
                f.write(format_row("Other Exceptions", stats["other_errors"]))
                f.write("\n")
                
    print(f"Saved error analysis report to {report_path}")

if __name__ == "__main__":
    main()
