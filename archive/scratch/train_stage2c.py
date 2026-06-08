"""
Stage 2A Fixed Training — Staged Run
=====================================
- Correct Qwen chat format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
- Labels: -100 for prompt tokens AND pad tokens
- eos = <|im_end|>  (id 151645)
- pad = <|endoftext|> (id 151643)
- bf16 AMP
- LR = 5e-5 with linear warmup 100 steps
- Grad clip = 0.5
- Stop immediately on NaN/Inf
- Save checkpoint every 500 steps + best non-NaN

Usage:
    python train\train_stage2a_fixed.py --steps 500
    python train\train_stage2a_fixed.py --steps 2000 --resume checkpoints\samat_next_350m_stage2a_fixed_step_500.pt
"""
import os
import sys
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM


# ── Tokenizer ────────────────────────────────────────────────────────────────
def load_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    assert tok.eos_token_id == 151645, f"Bad eos_token_id: {tok.eos_token_id}"
    assert tok.pad_token_id == 151643, f"Bad pad_token_id: {tok.pad_token_id}"
    return tok


def print_tokenizer_info(tok):
    print("=== Tokenizer ===")
    print(f"  eos_token : {repr(tok.eos_token):<28} id={tok.eos_token_id}")
    print(f"  pad_token : {repr(tok.pad_token):<28} id={tok.pad_token_id}")
    print(f"  vocab_size: {tok.vocab_size}  |  max_special_id: {max(tok.all_special_ids)}")
    print(f"  model vocab_size will be set from config")
    print("=================\n")


# ── Dataset ──────────────────────────────────────────────────────────────────
class FixedCodeDataset(Dataset):
    """
    Format per example:
      <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n   <- masked (-100)
      {target}<|im_end|>                                               <- train here
    """
    def __init__(self, data, tokenizer, max_len=512):
        self.tok = tokenizer
        self.max_len = max_len
        self.items = []
        self.skipped_count = 0
        self.skipped_examples = []
        self.valid_counts = []
        self.max_input_length = 0
        
        eos = tokenizer.eos_token  # <|im_end|>
        total_before = len(data)
        
        for idx, record in enumerate(data):
            p = record["prompt"].strip()
            t = record["target"].strip()
            prompt_text = f"<|im_start|>user\n{p}{eos}\n<|im_start|>assistant\n"
            target_text = f"{t}{eos}"
            
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            t_ids = tokenizer(target_text, add_special_tokens=False).input_ids
            
            input_ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + t_ids
            
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            valid_target_tokens = sum(1 for l in labels if l != -100)
            self.max_input_length = max(self.max_input_length, len(input_ids))
            
            if valid_target_tokens == 0:
                self.skipped_count += 1
                if len(self.skipped_examples) < 3:
                    self.skipped_examples.append({
                        "id": record.get("id", idx),
                        "source": record.get("source", "unknown"),
                        "prompt_len": len(p_ids)
                    })
                continue
                
            self.valid_counts.append(valid_target_tokens)
            self.items.append((input_ids, labels))
            
        print("\n=== Dataset Validation ===")
        print(f"Total before filtering: {total_before}")
        print(f"Total after filtering : {len(self.items)}")
        print(f"Skipped (target truncated): {self.skipped_count}")
        if self.valid_counts:
            print(f"Valid target tokens - min: {min(self.valid_counts)}, mean: {sum(self.valid_counts)/len(self.valid_counts):.1f}, max: {max(self.valid_counts)}")
        print(f"Max input length: {self.max_input_length}")
        if self.skipped_examples:
            print("First 3 skipped examples:")
            for s in self.skipped_examples:
                print(f"  - ID/Idx: {s['id']} | Source: {s['source']} | Prompt Len: {s['prompt_len']}")
        print("==========================\n")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        input_ids, labels = self.items[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch, pad_id):
    max_len = max(x[0].size(0) for x in batch)
    B = len(batch)
    inp = torch.full((B, max_len), pad_id, dtype=torch.long)
    lbl = torch.full((B, max_len), -100,   dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        L = ids.size(0)
        inp[i, :L] = ids
        lbl[i, :L] = labs
    return inp, lbl


# ── LR schedule: linear warmup then constant ──────────────────────────────────
def get_lr(step, warmup, base_lr):
    if step <= warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=1000,  help="Number of optimizer steps to run")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_step", type=int, default=0, help="Initial step count for resumed runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp32 for full numerical stability — model is 350M, fits comfortably in VRAM
    print(f"Device: {device}  |  Precision: float32 (AMP disabled)")
    print(f"Target steps: {args.steps}  |  Resume: {args.resume}\n")

    tok = load_tokenizer()
    print_tokenizer_info(tok)

    # Load dataset
    data_path = os.path.join(ROOT, "data", "stage2c_synthetic_curriculum.jsonl")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Dataset: {len(data)} examples loaded from {data_path}\n")

    dataset = FixedCodeDataset(data, tok, max_len=512)
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
        num_workers=0,
    )

    # Model
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    model  = SamatNextForCausalLM(config).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    else:
        print("Initializing from scratch (fresh random weights)")
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Hyperparameters
    BASE_LR      = 1e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 0.5
    GRAD_ACCUM   = 32     # effective batch = 32
    LOG_EVERY    = 20
    SAVE_EVERY   = 250

    optimizer = optim.AdamW(
        model.parameters(), lr=BASE_LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # Diagnostic hooks
    deltanet_max_abs   = []
    diffattn_max_logit = []
    diffattn_all_masked_count = []

    def make_dn_hook():
        def hook(module, inp, out):
            with torch.no_grad():
                deltanet_max_abs.append(out.float().abs().max().item())
        return hook

    def make_da_hook():
        # Capture pre-softmax logit stats by wrapping forward temporarily
        def hook(module, inp, out):
            # out is post-projection; capture hidden_states max abs as proxy
            with torch.no_grad():
                diffattn_max_logit.append(out.float().abs().max().item())
        return hook

    from samat_next.deltanet import GatedDeltaNet
    from samat_next.differential_attention import DifferentialAttention
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m.register_forward_hook(make_dn_hook())
        elif isinstance(m, DifferentialAttention):
            m.register_forward_hook(make_da_hook())

    LOG_EVERY = 10  # more frequent for smoke test
    SAVE_EVERY = 500

    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"),     exist_ok=True)

    log_data      = []
    best_loss     = float("inf")
    best_ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2c_best.pt")
    nan_happened  = False
    step_count    = args.start_step
    running_loss  = 0.0
    first_loss    = None

    model.train()
    print(f"=== Training for {args.steps} steps ===\n")

    data_iter = iter(loader)

    for batch_idx in range(args.steps * GRAD_ACCUM):
        try:
            inp, lbl = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, lbl  = next(data_iter)

        inp = inp.to(device)
        lbl = lbl.to(device)

        valid_labels = (lbl != -100).sum()
        if valid_labels == 0:
            print(f"Warning: Batch {batch_idx} has 0 valid labels! Skipping.")
            continue

        # LR schedule
        current_lr = get_lr(step_count + 1, WARMUP_STEPS, BASE_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        with torch.no_grad() if False else torch.enable_grad():
            pass  # placeholder
        deltanet_max_abs.clear()
        diffattn_max_logit.clear()
        diffattn_all_masked_count.clear()
        logits, _ = model(inp)
        sl = logits[..., :-1, :].contiguous()
        tl = lbl[..., 1:].contiguous()
        loss = loss_fct(sl.view(-1, sl.size(-1)), tl.view(-1)) / GRAD_ACCUM
        # capture max DeltaNet activation before clearing
        step_dn_max = max(deltanet_max_abs)    if deltanet_max_abs    else float('nan')
        step_da_max = max(diffattn_max_logit)  if diffattn_max_logit  else float('nan')
        step_da_amc = sum(diffattn_all_masked_count)

        # NaN check BEFORE backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nNaN/Inf in loss at batch {batch_idx}! Stopping immediately.")
            nan_happened = True
            break

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM == 0:

            # Gradient NaN check
            bad_grad = any(
                not torch.isfinite(p.grad).all()
                for p in model.parameters() if p.grad is not None
            )
            if bad_grad:
                print(f"\nNaN/Inf in gradients at step {step_count + 1}! Stopping.")
                nan_happened = True
                optimizer.zero_grad()
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            # Capture grad norm AFTER clipping
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.float().norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # Accumulate loss ONCE PER OPTIMIZER STEP (not per mini-batch)
            # loss was already divided by GRAD_ACCUM, so multiply back to get raw per-token loss
            step_loss = loss.item() * GRAD_ACCUM
            running_loss += step_loss
            if first_loss is None:
                first_loss = step_loss

            if step_count % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"Step {step_count:5d} | Loss: {avg:.4f} | LR: {current_lr:.2e} "
                      f"| GradNorm: {grad_norm:.3f} | DeltaNetMaxAbs: {step_dn_max:.3f} "
                      f"| DiffAttnMaxLogit: {step_da_max:.3f} | AllMaskedRows: {step_da_amc} "
                      f"| VRAM: {vram_gb:.2f}GB")
                log_data.append({"step": step_count, "loss": avg, "lr": current_lr,
                                  "grad_norm": grad_norm, "deltanet_max_abs": step_dn_max,
                                  "diffattn_max_logit": step_da_max})

                if avg < best_loss:
                    best_loss = avg
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"         -> Best checkpoint updated (loss={best_loss:.4f})")

                running_loss = 0.0

            if step_count % SAVE_EVERY == 0:
                ckpt = os.path.join(
                    ROOT, "checkpoints",
                    f"samat_next_350m_stage2c_step_{step_count}.pt"
                )
                torch.save(model.state_dict(), ckpt)
                print(f"         -> Checkpoint saved: {ckpt}")

            if step_count >= args.steps:
                break

    # Final save
    final_ckpt = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2c_latest.pt")
    torch.save(model.state_dict(), final_ckpt)

    # Persist log
    log_file = os.path.join(ROOT, "results", "stage2c_log.json")
    # Merge with existing log if resuming
    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                d = json.load(f)
                existing = d.get("log", [])
        except Exception:
            pass
    merged_log = existing + log_data

    with open(log_file, "w") as f:
        json.dump({
            "log": merged_log,
            "nan_happened": nan_happened,
            "best_loss": best_loss,
            "steps_completed": step_count,
        }, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Steps completed : {step_count}")
    print(f"Starting loss   : {first_loss:.4f}" if first_loss else "N/A")
    print(f"Final loss      : {log_data[-1]['loss']:.4f}" if log_data else "N/A")
    print(f"Best loss       : {best_loss:.4f}")
    print(f"NaN happened    : {nan_happened}")
    print(f"Latest ckpt     : {final_ckpt}")
    print(f"Best ckpt       : {best_ckpt_path}")

    # 3 sample generations
    TEST_PROMPTS = [
        "Write a Python function that adds two numbers.",
        "Write a Python function that checks if a number is even.",
        "Write a Python function that reverses a string.",
    ]
    eos_id  = tok.eos_token_id
    eos_tok = tok.eos_token
    print("\n=== 3 Sample Generations ===")
    model.eval()
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        input_text = f"<|im_start|>user\n{prompt}{eos_tok}\n<|im_start|>assistant\n"
        input_ids = tok(input_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        stopped = False
        with torch.no_grad():
            for _ in range(128):
                logits, _ = model(input_ids)
                next_id = torch.argmax(logits[0, -1, :]).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == eos_id:
                    stopped = True
                    break
        raw = tok.decode(input_ids[0], skip_special_tokens=False)
        marker = "<|im_start|>assistant\n"
        response = raw.split(marker)[-1] if marker in raw else raw
        print(f"\n[{i}] {prompt}")
        print(f"    Stopped at EOS: {stopped}")
        print(f"    Output:\n{response}")


if __name__ == "__main__":
    main()

