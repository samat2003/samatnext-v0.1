# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import time
import random
import math
import hashlib
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM

def get_file_sha256(filepath):
    if not filepath or not os.path.exists(filepath):
        return "none"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def get_git_dirty():
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return len(res.stdout.strip()) > 0
    except Exception:
        return False

class SimpleDataset(Dataset):
    def __init__(self, path, tok, max_seq):
        raw = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw.append(json.loads(line))
        self.examples = []
        self.tok = tok
        self.max_seq = max_seq
        self.eos_token = tok.eos_token
        
        skipped = 0
        for ex in raw:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{self.eos_token}\n<|im_start|>assistant\n"
            target_str = ex["target"] + self.eos_token
            full_str = prompt_str + target_str
            
            full_ids = tok(full_str, add_special_tokens=False,
                          max_length=max_seq, truncation=True).input_ids
            prompt_ids = tok(prompt_str, add_special_tokens=False).input_ids
            prompt_len = len(prompt_ids)
            
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]
            
            valid_target_tokens = sum(1 for l in labels if l != -100)
            if valid_target_tokens == 0:
                skipped += 1
                continue
                
            self.examples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "valid_tokens": valid_target_tokens
            })
        print(f"Loaded {len(self.examples)} examples from {path} ({skipped} skipped)")

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["input_ids"], ex["labels"], ex["valid_tokens"]

def collate(batch):
    ids_list, lbl_list, vtok_list = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    ids_pad = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    lbl_pad = torch.full((len(lbl_list), max_len), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
        ids_pad[i, :ids.size(0)] = ids
        lbl_pad[i, :lbl.size(0)] = lbl
    return ids_pad, lbl_pad, sum(vtok_list)

def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["samatnext", "transformer"])
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Apply smoke override
    if args.smoke:
        args.max_steps = 10
        args.eval_every = 5
        args.save_every = 5

    # Seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unified Training Wrapper | Model: {args.model.upper()} | Device: {device}")
    
    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Initialize model
    print(f"Loading config from {args.model_config}...")
    if args.model == "samatnext":
        config = SamatNextConfig.from_json(args.model_config)
        model = SamatNextForCausalLM(config).to(device)
    else:
        config = TransformerConfig.from_json(args.model_config)
        model = TransformerForCausalLM(config).to(device)

    # Print parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Set up datasets
    train_dataset = SimpleDataset(args.train, tok, args.context_length)
    val_dataset = SimpleDataset(args.val, tok, args.context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    # Optimizer and scheduler
    BASE_LR = 3e-6 if args.model == "transformer" else 3e-6 # use standard curriculum rates
    WARMUP = int(args.max_steps * 0.15) if not args.smoke else 2
    GRAD_CLIP = 0.5

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_schedule(s, WARMUP, args.max_steps))

    # Precision setup
    if args.dtype == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_amp = True
    elif args.dtype == "fp16" and torch.cuda.is_available():
        amp_dtype = torch.float16
        use_amp = True
    else:
        amp_dtype = torch.float32
        use_amp = False

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    opt_step = 0
    micro_batch_count = 0
    first_loss = None
    best_loss = float("inf")
    nan_detected = False
    log_rows = []
    
    data_iter = iter(train_loader)
    t0 = time.time()

    print(f"Starting training loop. Target steps: {args.max_steps}")
    
    while opt_step < args.max_steps:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_valid_tokens = 0

        for _ in range(args.grad_accum):
            try:
                ids, labels, valid_tokens = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                ids, labels, valid_tokens = next(data_iter)

            ids = ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss_scaled = loss / args.grad_accum

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
                
            accum_loss += loss.item() / args.grad_accum
            accum_valid_tokens += valid_tokens
            micro_batch_count += 1

        # Check NaN/Inf in loss
        if torch.isnan(torch.tensor(accum_loss)) or torch.isinf(torch.tensor(accum_loss)):
            print(f"Step {opt_step+1}: NaN/Inf detected in loss — stopping.")
            nan_detected = True
            break

        # Grad norm and clip
        if use_amp:
            scaler.unscale_(optimizer)
            
        gn_before = sum(p.grad.float().norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        gn_after = sum(p.grad.float().norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if first_loss is None:
            first_loss = accum_loss
            
        if accum_loss < best_loss:
            best_loss = accum_loss
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pt"))

        opt_step += 1

        # Periodic log
        if opt_step % 10 == 0 or opt_step == 1:
            elapsed = time.time() - t0
            vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"Step {opt_step:>4}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {lr_now:.2e} | GradNorm: {gn_before:.3f}/{gn_after:.3f} | VRAM: {vram:.0f}MB | Time: {elapsed:.0f}s")

        log_rows.append({
            "step": opt_step,
            "loss": accum_loss,
            "lr": lr_now,
            "grad_norm_pre": gn_before,
            "grad_norm_post": gn_after,
            "valid_tokens": accum_valid_tokens
        })

        if opt_step % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output, f"model_step_{opt_step}.pt"))

    # Training complete
    elapsed_total = time.time() - t0
    final_ckpt_path = os.path.join(args.output, "latest_model.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    
    # Save training log JSON
    with open(os.path.join(args.output, "training_log.json"), "w") as f:
        json.dump({
            "total_steps": opt_step,
            "first_loss": first_loss,
            "best_loss": best_loss,
            "final_loss": accum_loss,
            "nan_detected": nan_detected,
            "elapsed_seconds": elapsed_total,
            "log": log_rows
        }, f, indent=4)

    # Save Run Manifest
    print("Generating run manifest...")
    manifest = {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "model": args.model,
        "model_config_path": args.model_config,
        "model_config_hash": get_file_sha256(args.model_config),
        "tokenizer_path": args.tokenizer,
        "tokenizer_hash": "none" if "/" in args.tokenizer else get_file_sha256(args.tokenizer),
        "train_data_path": args.train,
        "train_data_hash": get_file_sha256(args.train),
        "val_data_path": args.val,
        "val_data_hash": get_file_sha256(args.val),
        "checkpoint_path": final_ckpt_path,
        "checkpoint_hash": get_file_sha256(final_ckpt_path),
        "seed": args.seed,
        "dtype": args.dtype,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "optimizer": "AdamW",
        "learning_rate": BASE_LR,
        "max_steps": args.max_steps,
        "wall_clock_time_seconds": elapsed_total,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "none",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_vram": torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0
    }
    
    with open(os.path.join(args.output, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Run complete. Manifest saved to {os.path.join(args.output, 'run_manifest.json')}")

if __name__ == "__main__":
    main()
