import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    # These environment variables are populated by torchrun or our fallback
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Use gloo on Windows, nccl on Linux
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def _init_weights(module):
    """
    Standard GPT-2 initialization, plus zero-init for residual projections
    to ensure the network starts as a stable identity function.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def _apply_zero_init(model):
    for name, param in model.named_parameters():
        if "o_proj.weight" in name or "down_proj.weight" in name:
            torch.nn.init.zeros_(param)

def cleanup_ddp():
    dist.destroy_process_group()

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rank, world_size, seq_len=2048):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.examples = []
        
        # ── Dataset 1: CodeFeedback (filtered to Python-only) ──
        if is_main_process_hack():
            print("Loading m-a-p/CodeFeedback-Filtered-Instruction...")
        ds_cf = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        
        python_keywords = ["python", "def ", "import ", "class ", "print(", ".py"]
        kept = 0
        skipped = 0
        for row in ds_cf:
            query = row.get("query", "")
            answer = row.get("answer", "")
            combined = (query + answer).lower()
            
            # Filter: keep only Python-related examples
            if not any(kw in combined for kw in python_keywords):
                skipped += 1
                continue
            
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > seq_len + 1:
                tokens = tokens[:seq_len + 1]
            self.examples.append(tokens)
            kept += 1
        
        if is_main_process_hack():
            print(f"  CodeFeedback: kept {kept}, skipped {skipped} non-Python")
        
        # ── Dataset 2: MBPP (algorithmic Python problems) ──
        if is_main_process_hack():
            print("Loading google-research-datasets/mbpp...")
        try:
            ds_mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
            mbpp_count = 0
            for row in ds_mbpp:
                text = row.get("text", "")
                code = row.get("code", "")
                tests = row.get("test_list", [])
                test_str = "\n".join(tests) if tests else ""
                
                prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n```python\n{code}\n```\n\nTest cases:\n```python\n{test_str}\n```<|im_end|>"
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > seq_len + 1:
                    tokens = tokens[:seq_len + 1]
                self.examples.append(tokens)
                mbpp_count += 1
            if is_main_process_hack():
                print(f"  MBPP: added {mbpp_count} examples")
        except Exception as e:
            if is_main_process_hack():
                print(f"  MBPP failed to load ({e}), skipping.")
        
        if world_size > 1:
            # Shard after combining both datasets
            self.examples = self.examples[rank::world_size]
        
        if is_main_process_hack():
            print(f"  TOTAL: {len(self.examples)} training examples loaded!")
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad to max length so batching works flawlessly
        pad_len = (self.seq_len + 1) - len(tokens)
        
        # If pad_token_id is not set, use eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        if pad_len > 0:
            tokens = tokens + [pad_id] * pad_len
            
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Mask out padding tokens from the loss calculation (-100 is ignored by CrossEntropyLoss)
        labels = [l if l != pad_id else -100 for l in labels]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def is_main_process_hack():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0

def train():
    # 1. Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    # Configuration
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "samat_next_v0_1.json")
    CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "finetune")
    PRETRAIN_CKPT = os.path.join(ROOT_DIR, "checkpoints", "pretrain", "step_1500.pt")
    
    if is_main_process:
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"Starting Multi-GPU Instruction Fine-Tuning with {world_size} GPUs.")

    # 2. Setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    config = SamatNextConfig.from_json(CONFIG_PATH)
    
    # Make sure vocab size matches
    config.vocab_size = len(tokenizer)
    
    model = SamatNextForCausalLM(config)
    
    # Apply stable initialization
    model.apply(_init_weights)
    _apply_zero_init(model)
    
    # Load Pre-trained Weights before moving to device!
    if is_main_process:
        print(f"Loading pre-trained foundational weights from {PRETRAIN_CKPT}...")
    checkpoint = torch.load(PRETRAIN_CKPT, map_location="cpu")
    # Our pretrain saved the model state dict directly for step_500/1500, or nested under 'model_state_dict'.
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    
    model = model.to(local_rank)
    
    # SlidingSpeed Integration: Torch Compile
    if os.environ.get("SLIDINGSPEED_TORCH_COMPILE") == "1":
        if is_main_process:
            print("SlidingSpeed: Enabling torch.compile()")
        model = torch.compile(model)
    
    # Wrap in DDP only if we have multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 3. Setup Optimizer & Scheduler
    use_fused = (os.environ.get("SLIDINGSPEED_FUSED_OPTIMIZER") == "1")
    if use_fused and is_main_process:
        print("SlidingSpeed: Enabling Fused AdamW")
    
    # Fine-tuning uses a much lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, fused=use_fused)
    
    # Standard LLM Warmup
    total_steps = 1500  # ~1 Epoch of CodeFeedback+MBPP with grad_accum=32
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=total_steps
    )
    
    # Start fresh for fine-tuning
    global_step = 0
    
    # 5. Training Loop
    seq_len = 2048
    grad_accum_steps = 32
    batch_size = 1
    
    # Setup data loading for fine-tuning
    dataset = ChatDataset(tokenizer, rank, world_size, seq_len=2048)
    
    # Standard PyTorch DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, # Shuffle fine-tuning data to prevent pattern memorization
        num_workers=0
    )
    
    model.train()
    step = 0
    running_loss = 0.0
    
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(local_rank)
        labels = labels.to(local_rank)
        
        # Forward pass with Automatic Mixed Precision (Bfloat16) to slash memory by 50%
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            
            if step == 0 and is_main_process:
                base_model = model.module if world_size > 1 else model
                print(f"[DEBUG] Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}, max: {logits.max().item():.4f}, min: {logits.min().item():.4f}")
                print(f"[DEBUG] Layer 0 o_proj.weight std: {base_model.model.layers[0].mixer.o_proj.weight.std().item():.6f}")
                print(f"[DEBUG] lm_head.weight std: {base_model.lm_head.weight.std().item():.6f}")
            
            # Calculate cross-entropy manually to avoid padding tokens since packed tokens are dense
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
            
            # Scale loss by grad accum steps
            loss = loss / grad_accum_steps
        
        # DDP optimization: only sync gradients on the last accumulation step
        is_last_accum_step = ((step + 1) % grad_accum_steps == 0)
        
        if is_last_accum_step:
            loss.backward()
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            if world_size > 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        # running_loss accumulates the scaled loss, which will sum up to the true average loss over the accum steps
        running_loss += loss.item()
        
        if is_last_accum_step:
            global_step = (step + 1) // grad_accum_steps
            
            if is_main_process:
                # Calculate tokens processed per step
                tokens_per_step = batch_size * seq_len * world_size * grad_accum_steps
                
                print(f"Step {global_step} | Loss: {running_loss:.4f} | Tokens/Step: {tokens_per_step}")
            
            running_loss = 0.0
            
            # Save Checkpoint every 500 steps
            if is_main_process and global_step % 500 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{global_step}.pt")
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
            # Early stopping after 500 local steps
            if global_step >= total_steps:
                if is_main_process:
                    print(f"Reached {total_steps} fine-tuning steps locally. Stopping!")
                    ckpt_path = os.path.join(CKPT_DIR, f"step_{total_steps}_final.pt")
                    torch.save((model.module if world_size > 1 else model).state_dict(), ckpt_path)
                break
                
        step += 1

    cleanup_ddp()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        with open("CRASH_REPORT.txt", "w") as f:
            traceback.print_exc(file=f)
        raise
