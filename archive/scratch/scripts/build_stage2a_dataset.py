import os
import json
import sys
import io
from datasets import load_dataset
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    out_path = os.path.join(ROOT, "data", "stage2a_code_pretrain.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    saved_examples = []
    seen_prompts = set()
    empty_target_count = 0
    duplicate_prompt_count = 0
    total_tokens_estimate = 0
    
    TARGET_COUNT = 50000
    
    def process_and_add(source, id_str, prompt, target):
        nonlocal empty_target_count, duplicate_prompt_count, total_tokens_estimate
        
        prompt = prompt.strip()
        target = target.strip()
        
        if not target:
            empty_target_count += 1
            return False
            
        if prompt in seen_prompts:
            duplicate_prompt_count += 1
            return False
            
        seen_prompts.add(prompt)
        
        # Estimate tokens
        # 1 char ~ 0.3 tokens roughly, but let's just use crude heuristic to save time
        # Or better, len.
        toks = (len(prompt) + len(target)) // 3
        total_tokens_estimate += toks
        
        saved_examples.append({
            "id": id_str,
            "source": source,
            "type": "generation",
            "prompt": prompt,
            "target": target
        })
        return True
        
    print("Loading datasets...")
    
    # 1. MBPP (google-research-datasets/mbpp)
    try:
        mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train+test+validation+prompt")
        for i, item in enumerate(mbpp):
            if len(saved_examples) >= TARGET_COUNT: break
            process_and_add(
                "Muennighoff/mbpp", 
                f"mbpp_{item['task_id']}", 
                item["prompt"], 
                item["code"]
            )
    except Exception as e:
        print(f"Error loading MBPP: {e}")
        
    print(f"Count after MBPP: {len(saved_examples)}")
    
    # 2. 18k Alpaca
    if len(saved_examples) < TARGET_COUNT:
        try:
            alpaca18k = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
            for i, item in enumerate(alpaca18k):
                if len(saved_examples) >= TARGET_COUNT: break
                prompt = item["instruction"]
                if item.get("input", ""):
                    prompt += "\n" + item["input"]
                process_and_add(
                    "iamtarun/python_code_instructions_18k_alpaca",
                    f"alpaca18k_{i}",
                    prompt,
                    item["output"]
                )
        except Exception as e:
            print(f"Error loading 18k Alpaca: {e}")
            
    print(f"Count after 18k Alpaca: {len(saved_examples)}")
    
    # 3. CodeAlpaca-20k
    if len(saved_examples) < TARGET_COUNT:
        try:
            ca20k = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            for i, item in enumerate(ca20k):
                if len(saved_examples) >= TARGET_COUNT: break
                prompt = item["instruction"]
                if item.get("input", ""):
                    prompt += "\n" + item["input"]
                process_and_add(
                    "sahil2801/CodeAlpaca-20k",
                    f"ca20k_{i}",
                    prompt,
                    item["output"]
                )
        except Exception as e:
            print(f"Error loading CodeAlpaca-20k: {e}")
            
    print(f"Count after CodeAlpaca-20k: {len(saved_examples)}")
    
    # 4. TokenBender/code_instructions_122k_alpaca_style
    if len(saved_examples) < TARGET_COUNT:
        try:
            tb122k = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
            for i, item in enumerate(tb122k):
                if len(saved_examples) >= TARGET_COUNT: break
                
                # Try to filter for Python if possible, but alpaca_style is mixed. 
                # At least filter by checking if it mentions python or if it's general code.
                text_to_check = (item["instruction"] + item["output"]).lower()
                if "python" in text_to_check or "def " in item["output"] or "import " in item["output"]:
                    prompt = item["instruction"]
                    if item.get("input", ""):
                        prompt += "\n" + item["input"]
                    process_and_add(
                        "TokenBender/code_instructions_122k_alpaca_style",
                        f"tb122k_{i}",
                        prompt,
                        item["output"]
                    )
        except Exception as e:
            print(f"Error loading 122k Alpaca: {e}")
            
    print(f"Count after 122k Alpaca: {len(saved_examples)}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in saved_examples:
            f.write(json.dumps(ex) + "\n")
            
    print("\n--- STAGE 2A DATASET REPORT ---")
    print(f"Total Examples: {len(saved_examples)}")
    print(f"Total Token Estimate (crude): ~{total_tokens_estimate:,}")
    print(f"Empty Target Count: {empty_target_count}")
    print(f"Duplicate Prompt Count: {duplicate_prompt_count}")
    
    print("\n--- FIRST 5 EXAMPLES ---")
    for i in range(min(5, len(saved_examples))):
        print(f"\nExample {i+1}:")
        print(json.dumps(saved_examples[i], indent=2))

if __name__ == "__main__":
    main()
