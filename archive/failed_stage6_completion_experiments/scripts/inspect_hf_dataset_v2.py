import os, ast
from datasets import load_dataset

ALLOWED_IMPORTS = {"math", "re", "itertools", "functools", "collections", "string", "typing"}

BANNED_KEYWORDS = [
    "HumanEval", "openai_humaneval", "canonical_solution", "check(candidate)", 
    "METADATA", "from humaneval", '"__main__"', "doctest", "unittest", "pytest",
    "```python", "eval(", "exec(", "subprocess", "os.system",
    "open(", "read(", "write(", "pathlib", "os.", 
    "requests", "django", "flask", "fastapi", "discord", "selenium", "rest_framework",
    "torch", "tensorflow", "keras", "sklearn", "pandas", "numpy", "scipy", "matplotlib"
]

def analyze_ast(tree, content):
    funcs = []
    classes = 0
    async_funcs = 0
    top_level_calls = 0
    has_print = False
    has_input = False
    has_assert = False
    
    # Check top level
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, ast.AsyncFunctionDef):
            async_funcs += 1
        elif isinstance(node, ast.FunctionDef):
            funcs.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # Exception for docstrings
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                top_level_calls += 1
        # Check imports at top level
        elif isinstance(node, ast.Import):
            for name in node.names:
                base_module = name.name.split('.')[0]
                if base_module not in ALLOWED_IMPORTS:
                    return f"Disallowed import: {base_module}"
        elif isinstance(node, ast.ImportFrom):
            base_module = node.module.split('.')[0] if node.module else ""
            if base_module not in ALLOWED_IMPORTS:
                return f"Disallowed importFrom: {base_module}"
                
    if classes > 0:
        return "Contains class definition"
    if async_funcs > 0:
        return "Contains async function"
    if len(funcs) != 1:
        return f"Contains {len(funcs)} functions (need exactly 1)"
    if top_level_calls > 0:
        return "Contains top-level execution code"
        
    func = funcs[0]
    
    if len(func.decorator_list) > 0:
        return "Contains decorators"
        
    has_return = False
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    has_print = True
                elif node.func.id == "input":
                    has_input = True
                elif node.func.id == "open":
                    return "File I/O (open) detected"
        elif isinstance(node, ast.Assert):
            has_assert = True
        elif isinstance(node, ast.Return):
            has_return = True
            
    if has_print:
        return "Contains print() statements"
    if has_input:
        return "Contains input() statements"
    if has_assert:
        return "Contains assert statements (likely tests)"
    if not has_return:
        return "No return statement (function must return a value)"
        
    # Check size
    lines = content.strip().split('\n')
    if len(lines) < 3:
        return "Function body too small (< 3 lines)"
    if len(lines) > 60:
        return "Function body too long (> 60 lines)"
        
    return None # No rejection reason!

def inspect_dataset():
    print("Loading dataset...")
    ds = load_dataset("jon-tow/starcoderdata-python-edu", split="train", streaming=True)
    
    total_sampled = 0
    parseable_count = 0
    accepted = []
    rejected = []
    
    # Track stats for report
    reason_counts = {}
    
    print("Streaming dataset...")
    for ex in ds:
        total_sampled += 1
        content = ex.get('content', '')
        
        rejection_reason = None
        
        # 1. Quick text keyword filters
        if len(content) > 1500: # Max ~512 tokens roughly
            rejection_reason = "Too long (>1500 chars)"
        else:
            for kw in BANNED_KEYWORDS:
                if kw in content:
                    rejection_reason = f"Contains banned keyword: {kw}"
                    break
                    
        # 2. AST parsing
        if rejection_reason is None:
            try:
                tree = ast.parse(content)
                parseable_count += 1
                ast_reason = analyze_ast(tree, content)
                if ast_reason:
                    rejection_reason = ast_reason
            except Exception as e:
                rejection_reason = f"SyntaxError/ParseFail: {type(e).__name__}"
                
        # 3. Categorize
        if rejection_reason is None:
            accepted.append(content)
            if len(accepted) % 10 == 0:
                print(f"Accepted {len(accepted)} / 100 ... (Processed {total_sampled})")
        else:
            if len(rejected) < 50:
                rejected.append((content, rejection_reason))
            reason_counts[rejection_reason.split(':')[0]] = reason_counts.get(rejection_reason.split(':')[0], 0) + 1
                
        if len(accepted) >= 100:
            break
            
        if total_sampled >= 50000:
            print("Reached 50,000 limit!")
            break
            
    print("Generating report...")
    report = f"""# Stage 6C Hugging Face Dataset Inspection Report (v2 - Strict Filtering)

## Overview
- **Dataset:** `jon-tow/starcoderdata-python-edu` (train split, streaming)
- **Rows Processed:** {total_sampled}
- **Accepted Examples:** {len(accepted)}
- **Overall Acceptance Rate:** {(len(accepted) / total_sampled) * 100:.2f}%
- **Parse Rate (among size-filtered):** {(parseable_count / total_sampled) * 100:.2f}%

## Exact Filtering Rules Used (v2)
1. **Length:** ≤ 1500 characters (fits 512 context easily).
2. **Structure:** Exactly 1 top-level function. No classes, no async, no decorators.
3. **Execution:** No top-level calls. No `__main__` or script execution logic.
4. **Imports:** NO external imports. Allowed imports ONLY: `math, re, itertools, functools, collections, string, typing`.
5. **Side Effects:** NO `print()`, NO `input()`, NO `open()`, NO `read()`, NO `write()`.
6. **Tests:** NO `assert` statements inside function. No pytest/unittest boilerplate.
7. **Return:** Function must contain a `return` statement.
8. **Lines:** Body must be between 3 and 60 lines.
9. **Banned Ecosystems:** ML (torch, pandas, sklearn, numpy), Web (django, requests, flask), OS/Pathing (os, pathlib).

## Top Rejection Reasons
"""
    # Sort rejection reasons by count
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:15]:
        report += f"- **{reason}**: {count}\n"

    report += "\n## 50 Accepted Examples After Filtering\n"
    for i, acc in enumerate(accepted[:50]):
        report += f"\n### Accepted {i+1}\n```python\n{acc.strip()}\n```\n"
        
    report += "\n## 50 Rejected Examples with Reasons\n"
    for i, (rej, rsn) in enumerate(rejected[:50]):
        report += f"\n### Rejected {i+1}\n**Reason:** {rsn}\n```python\n{rej.strip()[:300]}...\n```\n"

    with open("reports/stage6c_hf_dataset_inspection_v2.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Saved to reports/stage6c_hf_dataset_inspection_v2.md")

if __name__ == "__main__":
    inspect_dataset()
