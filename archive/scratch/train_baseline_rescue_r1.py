"""
Stage 5 Training: Baseline Transformer-350M Smoke Test
======================================================
Start from: Scratch (Random Init)
Data: data/stage5_teacher_distill.jsonl
Teacher: Qwen2.5-Coder-3B-Instruct
Hyperparams: lr=3e-6, warmup=300, grad_clip=0.5, 2000 optimizer updates, fp32
"""
import os, sys, json, random, math, time, re
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from baseline.config import TransformerConfig
from baseline.model import TransformerForCausalLM

TRAIN_FILE = os.path.join(ROOT, "data", "stage5_teacher_distill.jsonl")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
RESULTS    = os.path.join(ROOT, "results", "baseline_rescue_r1_log.json")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

BEST_CKPT  = os.path.join(CKPT_DIR, "transformer_350m_baseline_rescue_r1_best.pt")
STEP_CKPT  = os.path.join(CKPT_DIR, "transformer_350m_baseline_rescue_r1_step_{step}.pt")
BASE_CKPT  = os.path.join(CKPT_DIR, "transformer_350m_baseline_stage3_best.pt")

MAX_SEQ     = 384       
BATCH       = 1         
GRAD_ACCUM  = 16        
TOTAL_STEPS = 2000
LOG_EVERY   = 10
SAVE_EVERY  = 500
SAMPLE_EVERY = 200
BASE_LR     = 1e-5
WARMUP      = 300
GRAD_CLIP   = 0.5
SEED        = 42

class Stage5CurriculumDataset(Dataset):
    def __init__(self, path, tok, max_seq):
        raw = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.examples = []
        self.tok = tok
        self.max_seq = max_seq
        self.eos_token = tok.eos_token
        
        skipped = 0
        for ex in raw:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{self.eos_token}\n<|im_start|>assistant\n"
            target_str = ex["teacher_target"] + self.eos_token
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
                "valid_tokens": valid_target_tokens,
                "prompt": ex["prompt"],
                "function_name": ex.get("function_name"),
                "task_type": ex.get("task_type", "unknown"),
            })
        
        print(f"Loaded {len(self.examples)} examples ({skipped} skipped for zero valid tokens)")

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
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline Stage 5 Training | device={device}")
    print(f"Settings: lr={BASE_LR}, warmup={WARMUP}, grad_clip={GRAD_CLIP}")
    print(f"          total_steps={TOTAL_STEPS}, grad_accum={GRAD_ACCUM}, batch={BATCH}")
    print(f"          max_seq={MAX_SEQ}, fp32")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    print(f"Vocab size: {len(tok)}, Max tokenizer id: {max(tok.get_vocab().values())}")
    config = TransformerConfig(vocab_size=len(tok))
    model  = TransformerForCausalLM(config).to(device)
    
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
            
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    if not (330_000_000 <= param_count <= 370_000_000):
        print(f"ERROR: Parameter count {param_count:,} is outside 330M-370M range!")
        sys.exit(1)

    dataset = Stage5CurriculumDataset(TRAIN_FILE, tok, MAX_SEQ)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                         collate_fn=collate, drop_last=True)
                         
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))
    print(f"Loaded baseline Stage 3 checkpoint: {BASE_CKPT}")

    print("--- PRE-TRAINING SMOKE TEST ---")
    data_iter = iter(loader)
    dummy_ids, dummy_labels, _ = next(data_iter)
    dummy_ids, dummy_labels = dummy_ids.to(device), dummy_labels.to(device)
    model.eval()
    with torch.no_grad():
        d_logits, _ = model(dummy_ids)
        d_shift_logits = d_logits[..., :-1, :].contiguous()
        d_shift_labels = dummy_labels[..., 1:].contiguous()
        B, T, V = d_shift_logits.shape
        d_loss = torch.nn.functional.cross_entropy(d_shift_logits.view(B*T, V), d_shift_labels.view(B*T), ignore_index=-100)
    print(f"Logits shape: {d_logits.shape}")
    print(f"Initial loss is finite: {torch.isfinite(d_loss).item()} (Loss = {d_loss.item():.4f})")
    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print("-------------------------------")

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_schedule(s, WARMUP, TOTAL_STEPS))

    opt_step = 0
    mini_batch_count = 0
    first_loss = None
    best_loss = float("inf")
    nan_detected = False
    log_rows = []
    
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"Starting Baseline training from scratch: {TOTAL_STEPS} optimizer updates")
    print(f"{'='*70}\n")

    while opt_step < TOTAL_STEPS:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_valid_tokens = 0

        for _ in range(GRAD_ACCUM):
            try:
                ids, labels, valid_tokens = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                ids, labels, valid_tokens = next(data_iter)

            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            B, T_minus_1, V = shift_logits.shape
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(B * T_minus_1, V),
                shift_labels.view(B * T_minus_1),
                ignore_index=-100
            )

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM
            accum_valid_tokens += valid_tokens
            mini_batch_count += 1

        if torch.isnan(torch.tensor(accum_loss)) or torch.isinf(torch.tensor(accum_loss)):
            print(f"Step {opt_step+1}: NaN/Inf detected in loss — STOPPING IMMEDIATELY.")
            nan_detected = True
            break

        grad_has_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in model.parameters() if p.grad is not None
        )
        if grad_has_nan:
            print(f"Step {opt_step+1}: NaN/Inf detected in gradients — STOPPING IMMEDIATELY.")
            nan_detected = True
            break

        gn_before = sum(
            p.grad.float().norm().item()**2 
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        gn_after = sum(
            p.grad.float().norm().item()**2 
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        if first_loss is None:
            first_loss = accum_loss
        if accum_loss < best_loss:
            best_loss = accum_loss
            torch.save(model.state_dict(), BEST_CKPT)

        opt_step += 1

        if opt_step % LOG_EVERY == 0 or opt_step == 1:
            elapsed = time.time() - t0
            vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"OptStep {opt_step:>4}/{TOTAL_STEPS} | MBs: {mini_batch_count:>6} | "
                  f"Loss: {accum_loss:.4f} | LR: {lr_now:.2e} | "
                  f"GN (pre/post): {gn_before:.3f}/{gn_after:.3f} | "
                  f"ValidToks: {accum_valid_tokens} | "
                  f"VRAM: {vram:.0f}MB | "
                  f"Time: {elapsed:.0f}s")

        log_rows.append({
            "opt_step": opt_step,
            "micro_batches": mini_batch_count,
            "loss": accum_loss,
            "lr": lr_now,
            "gn_before": gn_before,
            "gn_after": gn_after,
            "valid_tokens": accum_valid_tokens,
            "nan_inf": False,
        })

        if opt_step % SAVE_EVERY == 0:
            path = STEP_CKPT.format(step=opt_step)
            torch.save(model.state_dict(), path)
            print(f"  >>> Checkpoint saved: {path}")

        if opt_step % SAMPLE_EVERY == 0:
            model.eval()
            print(f"\n  --- Sample generations at optimizer step {opt_step} ---")
            sample_indices = random.sample(range(len(dataset.examples)), min(3, len(dataset.examples)))
            for si in sample_indices:
                sample_ex = dataset.examples[si]
                prompt_str = f"<|im_start|>user\n{sample_ex['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
                inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                stopped = False
                with torch.no_grad():
                    for _ in range(192):
                        sl, _ = model(inp_ids)
                        nxt = torch.argmax(sl[0, -1, :]).item()
                        inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                        if nxt == tok.eos_token_id:
                            stopped = True
                            break
                raw = tok.decode(inp_ids[0], skip_special_tokens=False)
                gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").strip()
                gen_fn = (m.group(1) if (m := re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", gen)) else "NONE")
                exp_fn = sample_ex.get("function_name", "N/A")
                print(f"  [{sample_ex['task_type'].upper()}] Expected: {exp_fn} | Got: {gen_fn} | EOS: {stopped}")
                print(f"  Code: {gen[:150]}")
                print()
            model.train()

    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Stage 5 Baseline Training Complete")
    print(f"{'='*70}")
    print(f"Optimizer updates : {opt_step}")
    print(f"Micro-batches     : {mini_batch_count}")
    print(f"First loss        : {first_loss:.4f}" if first_loss else "First loss: N/A")
    print(f"Best loss         : {best_loss:.4f}")
    print(f"Final loss        : {accum_loss:.4f}")
    print(f"NaN/Inf detected  : {nan_detected}")
    print(f"Total time        : {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"Best checkpoint   : {BEST_CKPT}")

    with open(RESULTS, "w") as f:
        json.dump({
            "total_opt_steps": opt_step,
            "total_micro_batches": mini_batch_count,
            "first_loss": first_loss,
            "best_loss": best_loss,
            "final_loss": accum_loss,
            "nan_detected": nan_detected,
            "elapsed_seconds": elapsed_total,
            "log": log_rows
        }, f, indent=2)
    print(f"Training log saved: {RESULTS}")

if __name__ == "__main__":
    main()
