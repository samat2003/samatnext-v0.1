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
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

from scripts.eval_unseen import UNSEEN_PROMPTS, check_syntax, check_execution
from train.train_tiny_overfit import generate_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2_distill.pt")
    model = SamatNextForCausalLM(config).to(device)
    print(f"Loading checkpoint {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    # 1. Evaluate 20 unseen prompts
    print("Evaluating 20 unseen tiny prompts...")
    unseen_syntax_valid_count = 0
    unseen_test_pass_count = 0
    nan_inf_found = False
    
    samples = []
    
    for i, item in enumerate(UNSEEN_PROMPTS):
        instruction = item["instruction"]
        test_code = item["test"]
        
        generated = generate_text(model, tokenizer, instruction, device, max_new_tokens=150)
        
        prompt_str = f"<|user|>\n{instruction}\n<|assistant|>\n"
        if generated.startswith(prompt_str):
            assistant_response = generated[len(prompt_str):]
        else:
            assistant_response = generated.split("<|assistant|>\n")[-1] if "<|assistant|>\n" in generated else generated
            
        for token in ["<|user|>", "<|assistant|>", "<|end|>"]:
            assistant_response = assistant_response.replace(token, "")
        assistant_response = assistant_response.strip()
        
        is_syntax_valid = check_syntax(assistant_response)
        if is_syntax_valid:
            unseen_syntax_valid_count += 1
            
        is_exec_pass = check_execution(assistant_response, test_code) if is_syntax_valid else False
        if is_exec_pass:
            unseen_test_pass_count += 1
            
        res = {
            "instruction": instruction,
            "generated": assistant_response,
            "syntax_valid": is_syntax_valid,
            "test_pass": is_exec_pass
        }
        
        if i < 5:
            samples.append(res)
            
        # Check nan inf
        lm_text = f"<|user|>\n{instruction}\n<|assistant|>\n{assistant_response}"
        enc = tokenizer(lm_text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, v_logits = model(enc.input_ids)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_inf_found = True
                
    unseen_syntax_rate = unseen_syntax_valid_count / len(UNSEEN_PROMPTS)
    unseen_test_pass_rate = unseen_test_pass_count / len(UNSEEN_PROMPTS)
    
    # 2. Evaluate HumanEval (first 5)
    print("Evaluating first 5 HumanEval tasks...")
    he_test_pass_count = 0
    try:
        humaneval = load_dataset("openai_humaneval", split="test")
        he_items = list(humaneval)[:5]
        
        for item in he_items:
            instruction = item["prompt"]
            # Just generate directly as continuation
            generated = generate_text(model, tokenizer, instruction, device, max_new_tokens=150)
            
            prompt_str = f"<|user|>\n{instruction}\n<|assistant|>\n"
            if generated.startswith(prompt_str):
                assistant_response = generated[len(prompt_str):]
            else:
                assistant_response = generated.split("<|assistant|>\n")[-1] if "<|assistant|>\n" in generated else generated
                
            for token in ["<|user|>", "<|assistant|>", "<|end|>"]:
                assistant_response = assistant_response.replace(token, "")
            assistant_response = assistant_response.strip()
            
            full_code = item["prompt"] + assistant_response
            test_str = full_code + "\n" + item["test"] + f"\ncheck({item['entry_point']})"
            
            # exec test
            try:
                exec(test_str, {})
                he_test_pass_count += 1
            except:
                pass
    except Exception as e:
        print(f"HumanEval error: {e}")
        he_items = []
        
    he_score = he_test_pass_count / max(len(he_items), 1)
    
    # Read training logs to get start/final loss
    log_path = os.path.join(ROOT, "results", "stage2_distill_log.json")
    start_loss = 0.0
    final_loss = 0.0
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_data = json.load(f)
            if log_data:
                start_loss = log_data[0]["loss"]
                final_loss = log_data[-1]["loss"]
                
    summary = {
        "start_loss": start_loss,
        "final_loss": final_loss,
        "unseen_syntax_valid_rate": unseen_syntax_rate,
        "unseen_test_pass_rate": unseen_test_pass_rate,
        "humaneval_first5_score": he_score,
        "nan_inf_found": nan_inf_found,
        "samples": samples
    }
    
    out_path = os.path.join(ROOT, "results", "stage2_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n--- STAGE 2 EVALUATION REPORT ---")
    print(f"Starting LM Loss: {start_loss:.4f}")
    print(f"Final LM Loss: {final_loss:.4f}")
    print(f"Syntactically Valid Rate (20 prompts): {unseen_syntax_rate*100:.2f}%")
    print(f"Test Pass Rate (20 prompts): {unseen_test_pass_rate*100:.2f}%")
    print(f"HumanEval First 5 Score: {he_score*100:.2f}%")
    print(f"NaN/Inf Status: {nan_inf_found}")
    
    print("\n--- 5 Sample Generations (from 20 prompts) ---")
    for j, s in enumerate(samples):
        print(f"\nSample {j+1}: {s['instruction']}")
        print(f"Pass Tests: {s['test_pass']}")
        print(f"Code:\n{s['generated']}")

if __name__ == "__main__":
    main()
