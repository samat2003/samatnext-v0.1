import os
import sys
import json
import random
import torch
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datasets import load_dataset
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
    print("Loading datasets...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", split="train+validation+test")
        examples = list(ds)
        random.seed(42)
        random.shuffle(examples)
        
        # Take 800 MBPP tasks (which covers basic, string/list, and algo)
        mbpp_prompts = []
        for ex in examples[:800]:
            mbpp_prompts.append("Write a Python function to " + ex['text'])
            
    except Exception as e:
        print(f"Error loading MBPP: {e}")
        # fallback
        mbpp_prompts = [f"Write a Python function to solve algorithm {i}" for i in range(800)]
        
    # 100 Bug fix examples
    bug_fix_prompts = []
    for i in range(100):
        bug_fix_prompts.append(f"Fix the bug in this Python code: \ndef calculate_{i}(n):\n    return n / 0")

    print(f"Collected prompts: {len(mbpp_prompts)} MBPP, {len(bug_fix_prompts)} bug-fix.")
    
    # 100 Verifier examples
    verifier_examples = []
    for i in range(100):
        instruction = f"Write a Python function that returns the number {i}."
        correct = f"def get_number_{i}():\n    return {i}"
        wrong = f"def get_number_{i}():\n    return {i + 1}"
        verifier_examples.append({
            "instruction": instruction,
            "correct_solution": correct,
            "wrong_solution": wrong,
            "type": "verifier"
        })
        
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
    
    print("Generating targets...")
    all_gen_prompts = mbpp_prompts + bug_fix_prompts
    
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_path = os.path.join(ROOT, "data", "stage2_distill_dataset.jsonl")
    
    # Write to JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        # Write generation tasks
        for i, p in enumerate(all_gen_prompts):
            if i % 50 == 0:
                print(f"Generating {i}/{len(all_gen_prompts)}...")
            ans = generate_answer(model, tokenizer, p, device)
            record = {
                "instruction": p,
                "correct_solution": ans,
                "type": "generation"
            }
            f.write(json.dumps(record) + "\n")
            
        # Write verifier tasks
        for ex in verifier_examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Saved {len(all_gen_prompts) + len(verifier_examples)} examples to {out_path}")

if __name__ == "__main__":
    main()
