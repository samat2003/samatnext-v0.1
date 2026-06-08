import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import io

# Fix encoding issues on windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

class DistillDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        record = self.data[idx]
        prompt = f"<|user|>\n{record['prompt']}\n<|assistant|>\n"
        target = f"{record['target']}<|end|>"
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        input_ids = torch.cat([prompt_ids, target_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids])
        
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
            
        return input_ids, labels

def collate_fn(batch, pad_token_id):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, (inp, lbl) in enumerate(batch):
        input_ids[i, :len(inp)] = inp
        labels[i, :len(lbl)] = lbl
        
    return input_ids, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = os.path.join(ROOT, "data", "stage2_distill_dataset_clean.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    print(f"Loaded {len(data)} examples.")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Dataset and DataLoader
    dataset = DistillDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    
    # Initialize student model from scratch
    print("Initializing Samat-Next-Coder-350M from scratch...")
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    model = SamatNextForCausalLM(config)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss()
    
    grad_accum_steps = 8
    
    model.train()
    step_count = 0
    running_loss = 0.0
    
    log_data = []
    
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    
    print("Starting training...")
    
    for epoch in range(1):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, _ = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / grad_accum_steps
            
            loss.backward()
            
            # Check NaN/Inf
            nan_inf = False
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        nan_inf = True
                        break
            
            if nan_inf:
                print(f"Warning: NaN/Inf detected at step {step_count}. Skipping step.")
                optimizer.zero_grad()
                continue
                
            running_loss += loss.item() * grad_accum_steps
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                
                # Log loss
                if step_count % 20 == 0:
                    avg_loss = running_loss / 20
                    print(f"Step {step_count} | Loss: {avg_loss:.4f}")
                    log_data.append({"step": step_count, "loss": avg_loss})
                    running_loss = 0.0
                    
                # Save checkpoint
                if step_count % 200 == 0:
                    ckpt_path = os.path.join(ROOT, "checkpoints", f"samat_next_350m_stage2_distill_step_{step_count}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved intermediate checkpoint to {ckpt_path}")
                    
    # Final save
    final_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2_distill.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")
    
    with open(os.path.join(ROOT, "results", "stage2_distill_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)
        
    print("Training complete.")

if __name__ == "__main__":
    main()
