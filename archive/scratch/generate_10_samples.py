import os
import sys
import json
import torch
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def generate_answer(model, tokenizer, instruction, device):
    prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return assistant_response.strip()

def main():
    # 10 handcrafted high-quality coding tasks
    sample_prompts = [
        "Write a Python function reverse_string(s) that returns the reverse of string s.",
        "Write a Python function find_max(lst) that returns the maximum integer in a list lst. Return None if the list is empty.",
        "Write a Python function count_vowels(text) that returns the number of vowels (a, e, i, o, u) in the string text.",
        "Write a Python function flatten_list(nested_list) that takes a list of lists and returns a single flattened list.",
        "Write a Python function merge_dicts(d1, d2) that merges two dictionaries. If a key exists in both, sum their values.",
        "Write a Python function is_prime(n) that returns True if integer n is a prime number, and False otherwise.",
        "Write a Python function get_unique_elements(lst) that returns a new list containing only the unique elements of lst, preserving the original order.",
        "Fix the bug in this Python code that is supposed to calculate the average of a list:\n\ndef calculate_average(lst):\n    return sum(lst) / len(lst)\n# It fails when lst is empty. Fix it to return 0 if empty.",
        "Write a Python function that sorts a list of dictionaries by a specific key 'age' in descending order.",
        "Write a Python function fibonacci(n) that returns a list of the first n numbers in the Fibonacci sequence."
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Teacher model Qwen/Qwen2.5-Coder-3B-Instruct...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True
        )
    except Exception as e:
        print(f"Failed to load local files, trying without local_files_only: {e}")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
    model.eval()
    
    print("Generating 10 samples...")
    results = []
    for i, p in enumerate(sample_prompts):
        ans = generate_answer(model, tokenizer, p, device)
        results.append({
            "instruction": p,
            "correct_solution": ans,
            "type": "generation" if i < 7 or i > 7 else "bug-fix"
        })
        
    out_path = os.path.join(ROOT, "results", "10_samples.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved 10 samples to {out_path}")

if __name__ == "__main__":
    main()
