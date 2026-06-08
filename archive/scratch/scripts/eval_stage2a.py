import os
import sys
import io
import json
import traceback
import ast
import torch
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

from scripts.eval_unseen import UNSEEN_PROMPTS, check_syntax
from train.train_tiny_overfit import generate_text

def check_repetition(text):
    tokens = text.split()
    if len(tokens) < 10:
        return False
    c = Counter(tokens)
    # If the most common token makes up more than 30% of the output and output is reasonably long, it's repetitive
    if c.most_common(1)[0][1] > len(tokens) * 0.3:
        return True
    return False

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
    
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2a.pt")
    model = SamatNextForCausalLM(config).to(device)
    print(f"Loading checkpoint {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    # Evaluate 20 unseen prompts
    print("Evaluating 20 unseen tiny prompts...")
    unseen_syntax_valid_count = 0
    repetition_count = 0
    nan_inf_found = False
    
    samples = []
    
    for i, item in enumerate(UNSEEN_PROMPTS):
        instruction = item["instruction"]
        
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
            
        is_rep = check_repetition(assistant_response)
        if is_rep:
            repetition_count += 1
            
        res = {
            "instruction": instruction,
            "generated": assistant_response,
            "syntax_valid": is_syntax_valid,
            "is_repetitive": is_rep
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
    repetition_rate = repetition_count / len(UNSEEN_PROMPTS)
    
    # Read training logs to get start/final loss
    log_path = os.path.join(ROOT, "results", "stage2a_log.json")
    start_loss = 0.0
    final_loss = 0.0
    loss_curve = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_data = json.load(f)
            if log_data:
                start_loss = log_data[0]["loss"]
                final_loss = log_data[-1]["loss"]
                loss_curve = [item["loss"] for item in log_data]
                
    summary = {
        "start_loss": start_loss,
        "final_loss": final_loss,
        "loss_curve": loss_curve,
        "unseen_syntax_valid_rate": unseen_syntax_rate,
        "repetition_rate": repetition_rate,
        "nan_inf_found": nan_inf_found,
        "samples": samples
    }
    
    out_path = os.path.join(ROOT, "results", "stage2a_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n--- STAGE 2A EVALUATION REPORT ---")
    print(f"Starting LM Loss: {start_loss:.4f}")
    print(f"Final LM Loss: {final_loss:.4f}")
    print(f"Syntactically Valid Rate: {unseen_syntax_rate*100:.2f}%")
    print(f"Repetition Rate: {repetition_rate*100:.2f}%")
    print(f"NaN/Inf Status: {nan_inf_found}")
    
    print("\n--- 5 Sample Generations ---")
    for j, s in enumerate(samples):
        print(f"\nSample {j+1}: {s['instruction']}")
        print(f"Syntax Valid: {s['syntax_valid']} | Repetitive: {s['is_repetitive']}")
        print(f"Code:\n{s['generated']}")

if __name__ == "__main__":
    main()
