import os
import sys
import io
import json
import traceback
import ast
import torch
import torch.nn as nn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

# Use the same generation function
from train.train_tiny_overfit import generate_text

UNSEEN_PROMPTS = [
    {
        "instruction": "Write a Python function `merge_lists` that merges two lists.",
        "test": "assert merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]"
    },
    {
        "instruction": "Write a Python function `get_first_safe` that gets the first element of a list safely, returning None if empty.",
        "test": "assert get_first_safe([1]) == 1\nassert get_first_safe([]) is None"
    },
    {
        "instruction": "Write a Python function `flatten` that flattens a nested list by one level.",
        "test": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]"
    },
    {
        "instruction": "Write a Python function `absolute_value` that returns the absolute value of a number.",
        "test": "assert absolute_value(-5) == 5\nassert absolute_value(3) == 3"
    },
    {
        "instruction": "Write a Python function `contains_substring` that checks if a string contains a given substring.",
        "test": "assert contains_substring('hello world', 'world') is True\nassert contains_substring('hello', 'xyz') is False"
    },
    {
        "instruction": "Write a Python function `get_with_default` that gets a dictionary value with a default fallback.",
        "test": "assert get_with_default({'a': 1}, 'a', 0) == 1\nassert get_with_default({}, 'b', 0) == 0"
    },
    {
        "instruction": "Write a Python function `find_minimum` that finds the minimum in a list.",
        "test": "assert find_minimum([3, 1, 4]) == 1"
    },
    {
        "instruction": "Write a Python function `is_list_empty` that checks if a list is empty.",
        "test": "assert is_list_empty([]) is True\nassert is_list_empty([1]) is False"
    },
    {
        "instruction": "Write a Python function `square` that squares a number.",
        "test": "assert square(4) == 16"
    },
    {
        "instruction": "Write a Python function `cube` that cubes a number.",
        "test": "assert cube(3) == 27"
    },
    {
        "instruction": "Write a Python function `get_last_element` that gets the last element of a list.",
        "test": "assert get_last_element([1, 2, 3]) == 3"
    },
    {
        "instruction": "Write a Python function `find_intersection` that finds the intersection of two lists.",
        "test": "assert set(find_intersection([1, 2, 3], [2, 3, 4])) == {2, 3}"
    },
    {
        "instruction": "Write a Python function `to_uppercase` that converts a string to uppercase.",
        "test": "assert to_uppercase('hello') == 'HELLO'"
    },
    {
        "instruction": "Write a Python function `to_lowercase` that converts a string to lowercase.",
        "test": "assert to_lowercase('WORLD') == 'world'"
    },
    {
        "instruction": "Write a Python function `is_dict_empty` that checks if a dictionary is empty.",
        "test": "assert is_dict_empty({}) is True\nassert is_dict_empty({'a': 1}) is False"
    },
    {
        "instruction": "Write a Python function `repeat_string` that repeats a string n times.",
        "test": "assert repeat_string('ab', 3) == 'ababab'"
    },
    {
        "instruction": "Write a Python function `average` that computes the average of a list.",
        "test": "assert average([1, 2, 3, 4]) == 2.5"
    },
    {
        "instruction": "Write a Python function `all_positive` that checks if all elements in a list are positive.",
        "test": "assert all_positive([1, 2, 3]) is True\nassert all_positive([-1, 2]) is False"
    },
    {
        "instruction": "Write a Python function `get_dict_keys` that returns the keys of a dictionary as a list.",
        "test": "assert set(get_dict_keys({'a': 1, 'b': 2})) == {'a', 'b'}"
    },
    {
        "instruction": "Write a Python function `get_dict_values` that returns the values of a dictionary as a list.",
        "test": "assert set(get_dict_values({'a': 1, 'b': 2})) == {1, 2}"
    }
]

UNSEEN_PROMPTS_NATURAL = [
    {
        "instruction": "Write a Python function that merges two lists.",
        "test": "assert merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]"
    },
    {
        "instruction": "Write a Python function that gets the first element of a list safely, returning None if empty.",
        "test": "assert get_first_safe([1]) == 1\nassert get_first_safe([]) is None"
    },
    {
        "instruction": "Write a Python function that flattens a nested list by one level.",
        "test": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]"
    },
    {
        "instruction": "Write a Python function that returns the absolute value of a number.",
        "test": "assert absolute_value(-5) == 5\nassert absolute_value(3) == 3"
    },
    {
        "instruction": "Write a Python function that checks if a string contains a given substring.",
        "test": "assert contains_substring('hello world', 'world') is True\nassert contains_substring('hello', 'xyz') is False"
    },
    {
        "instruction": "Write a Python function that gets a dictionary value with a default fallback.",
        "test": "assert get_with_default({'a': 1}, 'a', 0) == 1\nassert get_with_default({}, 'b', 0) == 0"
    },
    {
        "instruction": "Write a Python function that finds the minimum in a list.",
        "test": "assert find_minimum([3, 1, 4]) == 1"
    },
    {
        "instruction": "Write a Python function that checks if a list is empty.",
        "test": "assert is_list_empty([]) is True\nassert is_list_empty([1]) is False"
    },
    {
        "instruction": "Write a Python function that squares a number.",
        "test": "assert square(4) == 16"
    },
    {
        "instruction": "Write a Python function that cubes a number.",
        "test": "assert cube(3) == 27"
    },
    {
        "instruction": "Write a Python function that gets the last element of a list.",
        "test": "assert get_last_element([1, 2, 3]) == 3"
    },
    {
        "instruction": "Write a Python function that finds the intersection of two lists.",
        "test": "assert set(find_intersection([1, 2, 3], [2, 3, 4])) == {2, 3}"
    },
    {
        "instruction": "Write a Python function that converts a string to uppercase.",
        "test": "assert to_uppercase('hello') == 'HELLO'"
    },
    {
        "instruction": "Write a Python function that converts a string to lowercase.",
        "test": "assert to_lowercase('WORLD') == 'world'"
    },
    {
        "instruction": "Write a Python function that checks if a dictionary is empty.",
        "test": "assert is_dict_empty({}) is True\nassert is_dict_empty({'a': 1}) is False"
    },
    {
        "instruction": "Write a Python function that repeats a string n times.",
        "test": "assert repeat_string('ab', 3) == 'ababab'"
    },
    {
        "instruction": "Write a Python function that computes the average of a list.",
        "test": "assert average([1, 2, 3, 4]) == 2.5"
    },
    {
        "instruction": "Write a Python function that checks if all elements in a list are positive.",
        "test": "assert all_positive([1, 2, 3]) is True\nassert all_positive([-1, 2]) is False"
    },
    {
        "instruction": "Write a Python function that returns the keys of a dictionary as a list.",
        "test": "assert set(get_dict_keys({'a': 1, 'b': 2})) == {'a', 'b'}"
    },
    {
        "instruction": "Write a Python function that returns the values of a dictionary as a list.",
        "test": "assert set(get_dict_values({'a': 1, 'b': 2})) == {1, 2}"
    }
]

def check_syntax(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def check_execution(code_str, test_code):
    try:
        local_scope = {}
        # Execute generated code
        exec(code_str, {}, local_scope)
        # Find the function name defined (just grabbing the first callable)
        funcs = [name for name, val in local_scope.items() if callable(val)]
        if not funcs:
            return False
            
        # The test cases expect specific names, but we can't guarantee the model generates them.
        # So we inject the generated function into the expected name using AST inspection if we can.
        # But for simplicity, we just exec the test directly in the same scope, assuming the model
        # named it somewhat naturally.
        # Actually, let's just make the test code flexible by finding the defined function and aliasing it to the expected name.
        # Let's extract the expected function name from the test string (e.g. 'merge_lists')
        import re
        match = re.search(r'assert (\w+)\(', test_code)
        if match:
            expected_name = match.group(1)
            # If expected name is not defined, map the first defined function to it
            if expected_name not in local_scope:
                local_scope[expected_name] = local_scope[funcs[0]]
        
        exec(test_code, {}, local_scope)
        return True
    except Exception:
        return False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_150m_tiny_overfit.pt")
    model = SamatNextForCausalLM(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    results = []
    
    syntax_valid_count = 0
    execution_pass_count = 0
    total_len = 0
    nan_inf_found = False
    
    samples = []
    
    print("Evaluating 20 unseen prompts...")
    for i, item in enumerate(UNSEEN_PROMPTS):
        instruction = item["instruction"]
        test_code = item["test"]
        
        # Generation
        generated = generate_text(model, tokenizer, instruction, device, max_new_tokens=100)
        
        # Strip prompt
        prompt_str = f"<|user|>\n{instruction}\n<|assistant|>\n"
        if generated.startswith(prompt_str):
            assistant_response = generated[len(prompt_str):]
        else:
            assistant_response = generated.split("<|assistant|>\n")[-1] if "<|assistant|>\n" in generated else generated
            
        for token in ["<|user|>", "<|assistant|>", "<|end|>"]:
            assistant_response = assistant_response.replace(token, "")
            
        assistant_response = assistant_response.strip()
        
        # Check syntax
        is_syntax_valid = check_syntax(assistant_response)
        if is_syntax_valid:
            syntax_valid_count += 1
            
        # Check execution
        is_exec_pass = check_execution(assistant_response, test_code) if is_syntax_valid else False
        if is_exec_pass:
            execution_pass_count += 1
            
        # Verifier Score
        lm_text = f"<|user|>\n{instruction}\n<|assistant|>\n{assistant_response}"
        enc = tokenizer(lm_text, return_tensors="pt").to(device)
        with torch.no_grad():
            _, v_logits = model(enc.input_ids)
            if torch.isnan(v_logits).any() or torch.isinf(v_logits).any():
                nan_inf_found = True
            v_score = torch.sigmoid(v_logits).item()
            
        total_len += len(assistant_response)
        
        res = {
            "instruction": instruction,
            "generated": assistant_response,
            "syntax_valid": is_syntax_valid,
            "test_pass": is_exec_pass,
            "verifier_score": v_score
        }
        results.append(res)
        
        if i < 5:
            samples.append(res)
            
    avg_len = total_len / len(UNSEEN_PROMPTS)
    
    summary = {
        "syntax_valid_rate": syntax_valid_count / len(UNSEEN_PROMPTS),
        "test_pass_rate": execution_pass_count / len(UNSEEN_PROMPTS),
        "average_generation_length": avg_len,
        "nan_inf_found": nan_inf_found,
        "results": results,
        "samples": samples
    }
    
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results", "stage1_6_unseen_tiny_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n--- RESULTS SUMMARY ---")
    print(f"Syntactically Valid Rate: {summary['syntax_valid_rate']*100:.2f}%")
    print(f"Test Pass Rate: {summary['test_pass_rate']*100:.2f}%")
    print(f"Average Generation Length: {avg_len:.1f} chars")
    print(f"NaN/Inf in Forward Pass: {nan_inf_found}")
    print("\n5 Sample Generations:")
    for j, s in enumerate(samples):
        print(f"\nSample {j+1}: {s['instruction']}")
        print(f"Verifier Score: {s['verifier_score']:.4f}")
        print(f"Pass Tests: {s['test_pass']}")
        print(f"Code:\n{s['generated']}")

if __name__ == "__main__":
    main()
