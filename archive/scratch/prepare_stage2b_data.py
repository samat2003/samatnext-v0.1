import os
import json
import re
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "data", "stage2a_code_pretrain.jsonl")
OUTPUT_FILE = os.path.join(ROOT, "data", "stage2b_hq_data.jsonl")
MAX_EXAMPLES = 10000

def has_syntax_error(code_str):
    try:
        ast.parse(code_str)
        return False
    except SyntaxError:
        return True

def filter_dataset():
    good_examples = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["prompt"]
            target = data["target"]
            
            # 1. Syntax check
            if has_syntax_error(target):
                continue
            
            # 2. Target must start with def or contain a function definition
            match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", target)
            if not match:
                continue
            
            func_name = match.group(1)
            
            # 3. Print-only / No-return logic
            has_return = "return " in target
            has_print = "print(" in target
            prompt_lower = prompt.lower()
            asks_for_print = "print" in prompt_lower or "display" in prompt_lower or "show" in prompt_lower
            asks_for_mutation = "mutate" in prompt_lower or "modify" in prompt_lower or "append" in prompt_lower or "update" in prompt_lower
            
            if not has_return:
                if not (asks_for_print or asks_for_mutation):
                    continue
            
            if has_print and not has_return:
                if not asks_for_print:
                    continue
                    
            # 4. Skip extremely short/generic bodies unless explicitly asked
            generic_bodies = ["return lst", "return string", "return True", "return False", "return []", "return None", "return 0", "return 1", "pass"]
            body_lines = [l.strip() for l in target.split("\n") if l.strip()]
            if len(body_lines) <= 2:
                # the body is just 1 line (plus the def line)
                if any(body_lines[-1] == g for g in generic_bodies):
                    # unless prompt asks for something incredibly simple, skip
                    if "always return" not in prompt_lower and "empty" not in prompt_lower and "true" not in prompt_lower and "false" not in prompt_lower:
                        continue
            
            # 5. Function-name alignment (explicit name injection)
            if func_name not in prompt:
                prompt += f" Name the function {func_name}."
                data["prompt"] = prompt
                    
            good_examples.append(data)
            if len(good_examples) >= MAX_EXAMPLES:
                break

    print(f"Filtered {len(good_examples)} high-quality examples.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for ex in good_examples:
            out.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    filter_dataset()
