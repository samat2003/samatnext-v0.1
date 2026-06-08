import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

class Stage2ADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
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
            # truncate from right to fit within max_len
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
    
    # Check bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using AMP dtype: {dtype}")
    
    # Load dataset
    data_path = os.path.join(ROOT, "data", "stage2a_code_pretrain.jsonl")
    print("Loading dataset...")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} examples.")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = Stage2ADataset(data, tokenizer, max_len=512)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    
    print("Initializing Samat-Next-Coder-350M from scratch...")
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    model = SamatNextForCausalLM(config)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss()
    
    # Optionally use GradScaler if using float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    
    grad_accum_steps = 32
    epochs = 2
    
    model.train()
    step_count = 0
    running_loss = 0.0
    log_data = []
    
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    
    print("Starting training...")
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            nan_inf = False
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        nan_inf = True
                        break
            
            if nan_inf:
                print(f"Warning: NaN/Inf detected at batch {batch_idx}. Skipping step.")
                optimizer.zero_grad()
                continue
                
            running_loss += loss.item() * grad_accum_steps
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                step_count += 1
                
                if step_count % 20 == 0:
                    avg_loss = running_loss / 20
                    print(f"Step {step_count} | Loss: {avg_loss:.4f}")
                    log_data.append({"step": step_count, "loss": avg_loss})
                    running_loss = 0.0
                    
                if step_count % 500 == 0:
                    ckpt_path = os.path.join(ROOT, "checkpoints", f"samat_next_350m_stage2a_step_{step_count}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved intermediate checkpoint to {ckpt_path}")
                    
    final_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2a.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")
    
    with open(os.path.join(ROOT, "results", "stage2a_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)
        
    print("Training complete.")

if __name__ == "__main__":
    main()
