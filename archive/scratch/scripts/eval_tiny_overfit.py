import os
import sys
import io
import torch
import torch.nn as nn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from train.losses import compute_lm_loss, compute_verifier_loss

# Import dataset generation and generate_text from the training script
from train.train_tiny_overfit import get_dataset, generate_text

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, dataset, tokenizer, device, prefix=""):
    model.eval()
    
    total_lm_loss = 0.0
    total_verifier_loss = 0.0
    raw_exact_matches = 0
    norm_exact_matches = 0
    contains_target_count = 0
    nan_inf_found = False
    
    verifier_correct_preds = []
    verifier_wrong_preds = []
    
    print(f"\n--- {prefix} Evaluation ---")
    
    with torch.no_grad():
        for i, ex in enumerate(dataset):
            # 1. Evaluate LM loss & generation
            lm_text = f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['correct_solution']}"
            lm_encodings = tokenizer(lm_text, return_tensors="pt").to(device)
            lm_input_ids = lm_encodings.input_ids
            
            lm_logits, _ = model(lm_input_ids)
            lm_loss = compute_lm_loss(lm_logits, lm_input_ids)
            total_lm_loss += lm_loss.item()
            
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                nan_inf_found = True
            
            # Generation exact match
            generated = generate_text(model, tokenizer, ex['instruction'], device, max_new_tokens=50)
            
            # 1. Strip the prompt from generated output
            prompt_str = f"<|user|>\n{ex['instruction']}\n<|assistant|>\n"
            if generated.startswith(prompt_str):
                assistant_response = generated[len(prompt_str):]
            else:
                assistant_response = generated.split("<|assistant|>\n")[-1] if "<|assistant|>\n" in generated else generated
                
            # 2. Remove special tokens
            for token in ["<|user|>", "<|assistant|>", "<|end|>"]:
                assistant_response = assistant_response.replace(token, "")
                
            clean_gen_raw = assistant_response.strip()
            clean_target_raw = ex['correct_solution'].strip()
            
            # 3. Normalize whitespace lightly
            clean_gen_norm = " ".join(clean_gen_raw.split())
            clean_target_norm = " ".join(clean_target_raw.split())
            
            if clean_gen_raw == clean_target_raw:
                raw_exact_matches += 1
                
            if clean_gen_norm == clean_target_norm:
                norm_exact_matches += 1
                
            if clean_target_norm in clean_gen_norm:
                contains_target_count += 1
                
            if i < 3:  # Print first 3 examples
                print(f"Sample {i+1} Instruction: {ex['instruction']}")
                print(f"Target: {clean_target_raw}")
                clean_gen_safe = clean_gen_raw.encode('utf-8', errors='replace').decode('utf-8')
                print(f"Generated (parsed): {clean_gen_safe}")
                print("-" * 20)
                
            # 2. Evaluate Verifier (1 logit per sequence)
            # Correct example
            correct_enc = tokenizer(lm_text, return_tensors="pt").to(device)
            _, v_logits_corr = model(correct_enc.input_ids)
            v_loss_corr = compute_verifier_loss(v_logits_corr, torch.tensor([1.0], device=device))
            total_verifier_loss += v_loss_corr.item()
            verifier_correct_preds.append(torch.sigmoid(v_logits_corr).item())
            
            # Wrong example
            wrong_text = f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['wrong_solution']}"
            wrong_enc = tokenizer(wrong_text, return_tensors="pt").to(device)
            _, v_logits_wrong = model(wrong_enc.input_ids)
            v_loss_wrong = compute_verifier_loss(v_logits_wrong, torch.tensor([0.0], device=device))
            total_verifier_loss += v_loss_wrong.item()
            verifier_wrong_preds.append(torch.sigmoid(v_logits_wrong).item())
            
    avg_lm_loss = total_lm_loss / len(dataset)
    avg_verifier_loss = total_verifier_loss / (2 * len(dataset))
    raw_em_rate = raw_exact_matches / len(dataset)
    norm_em_rate = norm_exact_matches / len(dataset)
    contains_rate = contains_target_count / len(dataset)
    
    avg_v_corr = sum(verifier_correct_preds) / len(verifier_correct_preds)
    avg_v_wrong = sum(verifier_wrong_preds) / len(verifier_wrong_preds)
    
    print(f"LM Loss: {avg_lm_loss:.4f}")
    print(f"Verifier Loss: {avg_verifier_loss:.4f}")
    print(f"Raw Exact Match Rate: {raw_em_rate*100:.2f}%")
    print(f"Normalized Exact Match Rate: {norm_em_rate*100:.2f}%")
    print(f"Contains-Target Rate: {contains_rate*100:.2f}%")
    print(f"Average Verifier Sigmoid (Correct Sol): {avg_v_corr:.4f} (Expected close to 1)")
    print(f"Average Verifier Sigmoid (Wrong Sol): {avg_v_wrong:.4f} (Expected close to 0)")
    print(f"NaN/Inf in Loss: {nan_inf_found}")
    
    return {
        "lm_loss": avg_lm_loss,
        "verifier_loss": avg_verifier_loss,
        "raw_em_rate": raw_em_rate,
        "norm_em_rate": norm_em_rate,
        "contains_rate": contains_rate,
        "nan_inf": nan_inf_found,
        "v_corr": avg_v_corr,
        "v_wrong": avg_v_wrong
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = get_dataset()
    unique_dataset = []
    seen = set()
    for ex in dataset:
        if ex['instruction'] not in seen:
            unique_dataset.append(ex)
            seen.add(ex['instruction'])
            
    print(f"Loaded {len(unique_dataset)} unique base examples for evaluation.")
    
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    
    # Untrained Model
    print("\nInitializing untrained model...")
    model_untrained = SamatNextForCausalLM(config).to(device)
    for module in model_untrained.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    num_params = count_parameters(model_untrained)
    print(f"Parameter Count: {num_params:,}")
    
    res_before = evaluate_model(model_untrained, unique_dataset, tokenizer, device, prefix="BEFORE (Untrained)")
    
    del model_untrained
    
    # Trained Model
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_150m_tiny_overfit.pt")
    print(f"\nLoading trained model from {ckpt_path}...")
    model_trained = SamatNextForCausalLM(config).to(device)
    model_trained.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    res_after = evaluate_model(model_trained, unique_dataset, tokenizer, device, prefix="AFTER (Trained)")
    
    # Print Final Summary
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(f"1. Checkpoint Path: {ckpt_path}")
    print(f"2. Parameter Count: {num_params:,}")
    print(f"3. Training LM Loss Before -> After: {res_before['lm_loss']:.4f} -> {res_after['lm_loss']:.4f}")
    print(f"4. Raw Exact-Match Rate on Overfit Set: {res_after['raw_em_rate']*100:.2f}%")
    print(f"   Normalized Exact-Match Rate: {res_after['norm_em_rate']*100:.2f}%")
    print(f"   Contains-Target Rate: {res_after['contains_rate']*100:.2f}%")
    print(f"5. Verifier Average Prob (Correct vs Wrong) Before: {res_before['v_corr']:.4f} vs {res_before['v_wrong']:.4f}")
    print(f"6. Verifier Average Prob (Correct vs Wrong) After : {res_after['v_corr']:.4f} vs {res_after['v_wrong']:.4f}")
    print(f"7. NaN/Inf Checks (Before/After): {res_before['nan_inf']} / {res_after['nan_inf']}")

if __name__ == "__main__":
    main()
