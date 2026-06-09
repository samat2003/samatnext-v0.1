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
import wandb

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

def setup_ddp():
    # These environment variables are populated by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def stream_and_pack_dataset(tokenizer, rank, world_size, seq_len=2048, batch_size=8):
    """
    Streams python-edu from HuggingFace, shards it across GPUs,
    and packs tokens up to seq_len for maximum efficiency.
    """
    # Stream the dataset, sharding it so each GPU sees unique data
    ds = load_dataset(
        "codeparrot/codeparrot-clean", 
        split="train", 
        streaming=True
    ).shard(num_shards=world_size, index=rank)
    
    buffer = []
    
    for row in ds:
        text = row["content"]
        # Tokenize and append the EOS token as a natural separator
        tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        buffer.extend(tokens)
        
        # Yield batches of exactly `seq_len`
        while len(buffer) >= (seq_len + 1) * batch_size:
            input_ids = []
            labels = []
            
            for i in range(batch_size):
                chunk = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]  # Keep overlap for next-token prediction shift
                
                input_ids.append(chunk[:-1])
                labels.append(chunk[1:])
                
            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )

def train():
    # 1. Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    # Configuration
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "samat_next_v0_1.json")
    CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "pretrain")
    if is_main_process:
        os.makedirs(CKPT_DIR, exist_ok=True)
        wandb.init(project="samatnext-pretrain", name="python-edu-14B")
        print(f"Starting Multi-GPU Pre-training with {world_size} GPUs.")

    # 2. Setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    config = SamatNextConfig.from_json(CONFIG_PATH)
    
    # Make sure vocab size matches
    config.vocab_size = len(tokenizer)
    
    model = SamatNextForCausalLM(config).to(local_rank)
    
    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # 4. Training Loop
    batch_size = 4  # Micro-batch size per GPU. Global = batch_size * world_size * grad_accum
    seq_len = 2048
    grad_accum_steps = 8
    
    data_stream = stream_and_pack_dataset(tokenizer, rank, world_size, seq_len, batch_size)
    
    model.train()
    step = 0
    running_loss = 0.0
    
    for input_ids, labels in data_stream:
        input_ids = input_ids.to(local_rank)
        labels = labels.to(local_rank)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Calculate cross-entropy manually to avoid padding tokens since packed tokens are dense
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
        
        # Scale loss by grad accum steps
        loss = loss / grad_accum_steps
        
        # DDP optimization: only sync gradients on the last accumulation step
        is_last_accum_step = ((step + 1) % grad_accum_steps == 0)
        
        if is_last_accum_step:
            loss.backward()
        else:
            with model.no_sync():
                loss.backward()
        
        running_loss += loss.item() * grad_accum_steps
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            global_step = (step + 1) // grad_accum_steps
            
            if is_main_process:
                # Calculate tokens processed per step
                tokens_per_step = batch_size * seq_len * world_size * grad_accum_steps
                
                print(f"Step {global_step} | Loss: {running_loss:.4f} | Tokens/Step: {tokens_per_step}")
                wandb.log({
                    "loss": running_loss,
                    "global_step": global_step,
                    "tokens_processed": global_step * tokens_per_step
                })
            
            running_loss = 0.0
            
            # Save Checkpoint every 5000 steps
            if is_main_process and global_step % 5000 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{global_step}.pt")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
            # Early stopping after ~24 hours on a single L4 GPU (45,000 global steps = ~2.9B tokens)
            if global_step >= 45000:
                if is_main_process:
                    print("Reached 45,000 steps (~24 hours). Stopping early for quality verification!")
                    ckpt_path = os.path.join(CKPT_DIR, "step_45000_final.pt")
                    torch.save(model.module.state_dict(), ckpt_path)
                break
                
        step += 1

    cleanup_ddp()

if __name__ == "__main__":
    train()
