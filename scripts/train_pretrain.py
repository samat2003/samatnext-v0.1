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

class StreamingPackedDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, rank, world_size, seq_len=2048, batch_size=8):
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        row_idx = 0
        buffer = []
        
        while True:
            try:
                ds = load_dataset(
                    "codeparrot/codeparrot-clean", 
                    split="train", 
                    streaming=True
                ).shard(num_shards=self.world_size, index=self.rank)
                
                ds_iter = iter(ds)
                
                # Fast-forward to the exact file we were on before the connection dropped
                for _ in range(row_idx):
                    next(ds_iter)
                
                for row in ds_iter:
                    text = row["content"]
                    tokens = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                    buffer.extend(tokens)
                    
                    while len(buffer) >= (self.seq_len + 1) * self.batch_size:
                        input_ids = []
                        labels = []
                        for i in range(self.batch_size):
                            chunk = buffer[:self.seq_len + 1]
                            buffer = buffer[self.seq_len:] 
                            input_ids.append(chunk[:-1])
                            labels.append(chunk[1:])
                        
                        yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
                    
                    row_idx += 1
            except StopIteration:
                break # Reached the end of the entire dataset
            except Exception as e:
                print(f"\n[Network] HuggingFace connection dropped ({e}). Reconnecting...\n")
                import time
                time.sleep(5)
                continue

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
        print(f"Starting Multi-GPU Pre-training with {world_size} GPUs.")

    # 2. Setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    config = SamatNextConfig.from_json(CONFIG_PATH)
    
    # Make sure vocab size matches
    config.vocab_size = len(tokenizer)
    
    model = SamatNextForCausalLM(config)
    
    # Apply stable initialization
    model.apply(_init_weights)
    _apply_zero_init(model)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=use_fused)
    
    # Standard LLM Warmup (2000 steps warmup, then cosine decay)
    total_steps = 45000  # Our ~24 hour early stopping target
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=2000, 
        num_training_steps=total_steps
    )
    
    # 4. Auto-Resume Checkpoint Logic
    global_step = 0
    if os.path.exists(CKPT_DIR):
        import glob
        import re
        ckpts = glob.glob(os.path.join(CKPT_DIR, "step_*.pt"))
        if ckpts:
            # Find latest checkpoint
            def get_step(p):
                match = re.search(r'step_(\d+)', p)
                return int(match.group(1)) if match else -1
            latest_ckpt = max(ckpts, key=get_step)
            global_step = get_step(latest_ckpt)
            
            if is_main_process:
                print(f"Resuming from checkpoint: {latest_ckpt} at step {global_step}")
                
            checkpoint = torch.load(latest_ckpt, map_location=f"cuda:{local_rank}")
            (model.module if world_size > 1 else model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            del checkpoint
    
    # 5. Training Loop
    batch_size = 1  # Micro-batch size per GPU. Global = batch_size    # Setup data loading
    dataset = StreamingPackedDataset(tokenizer, rank, world_size, seq_len=2048, batch_size=1)
    
    # We use num_workers=0 because HuggingFace streaming httpx client crashes on Windows spawn multiprocessing
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=None,
        num_workers=0
    )
    data_stream = iter(dataloader)
    
    model.train()
    step = global_step * 32
    running_loss = 0.0
    seq_len = 2048
    grad_accum_steps = 32
    
    # Fast-forward dataloader
    if is_main_process and global_step > 0:
        print(f"Fast-forwarding dataset to step {global_step}...")
    for _ in range(step):
        next(data_stream)
    
    for input_ids, labels in data_stream:
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
            if global_step >= 4100:
                if is_main_process:
                    print("Reached 4100 steps locally. Stopping early for quality verification!")
                    ckpt_path = os.path.join(CKPT_DIR, "step_4100_final.pt")
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
