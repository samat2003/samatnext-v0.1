"""
Stage 4B Training: Explicit Identifier Copy Curriculum
=======================================================
Start from: checkpoints/samat_next_350m_stage4_best.pt
Data: data/stage4b_name_copy_train.jsonl
Hyperparams: lr=5e-7, warmup=300, grad_clip=0.5, 1000 steps, fp32
"""
import os, sys, json, random, math, time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

TRAIN_FILE = os.path.join(ROOT, "data", "stage4b_name_copy_train.jsonl")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
RESULTS    = os.path.join(ROOT, "results", "stage4b_log.json")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

BASE_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage4_best.pt")
BEST_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage4b_best.pt")
STEP_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage4b_step_{step}.pt")

MAX_SEQ = 256
BATCH   = 4
GRAD_ACCUM = 16    # effective batch = 64
TOTAL_STEPS = 1_000
LOG_EVERY   = 10
SAVE_EVERY  = 250
BASE_LR     = 5e-7
WARMUP      = 300
GRAD_CLIP   = 0.5
SEED        = 42

class CodeDataset(Dataset):
    def __init__(self, path, tok, max_seq, eos_tok):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok     = tok
        self.max_seq = max_seq
        self.eos_tok = eos_tok

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = f"<|im_start|>user\n{ex['prompt']}{self.eos_tok}\n<|im_start|>assistant\n"
        full   = prompt + ex["target"] + self.eos_tok
        ids = self.tok(full, add_special_tokens=False,
                       max_length=self.max_seq, truncation=True).input_ids
        ids = torch.tensor(ids, dtype=torch.long)
        # Labels: mask the prompt tokens
        prompt_len = len(self.tok(prompt, add_special_tokens=False).input_ids)
        labels = ids.clone()
        labels[:prompt_len] = -100
        return ids, labels

def collate(batch):
    ids_list, lbl_list = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    ids_pad = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    lbl_pad = torch.full((len(lbl_list), max_len), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
        ids_pad[i, :ids.size(0)] = ids
        lbl_pad[i, :lbl.size(0)] = lbl
    return ids_pad, lbl_pad

def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 4B Training | device={device}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not os.path.exists(BASE_CKPT):
        print(f"ERROR: Base checkpoint not found: {BASE_CKPT}")
        sys.exit(1)
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))
    print(f"Loaded from: {BASE_CKPT}")

    dataset = CodeDataset(TRAIN_FILE, tok, MAX_SEQ, tok.eos_token)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                         collate_fn=collate, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_schedule(s, WARMUP, TOTAL_STEPS))

    # For logging:
    opt_step = 0
    mini_batch_count = 0
    data_iter = iter(loader)
    first_loss = None
    best_loss = float("inf")
    nan_detected = False
    log_rows = []

    while opt_step < TOTAL_STEPS:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                ids, labels = next(data_iter)

            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            
            # CRITICAL FIX: Shift logits and labels for next-token prediction
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
            mini_batch_count += 1

        if torch.isnan(torch.tensor(accum_loss)) or torch.isinf(torch.tensor(accum_loss)):
            print(f"Step {opt_step}: NaN/Inf detected — stopping.")
            nan_detected = True
            break

        # Grad norm before clipping
        gn_before = sum(p.grad.float().norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Grad norm after clipping
        gn_after = sum(p.grad.float().norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        optimizer.step()
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        if first_loss is None: first_loss = accum_loss
        if accum_loss < best_loss:
            best_loss = accum_loss
            torch.save(model.state_dict(), BEST_CKPT)

        if (opt_step + 1) % LOG_EVERY == 0 or opt_step == 0:
            print(f"Step {opt_step+1:>4}/{TOTAL_STEPS} | MBs: {mini_batch_count:>5} | "
                  f"Loss: {accum_loss:.4f} | LR: {lr_now:.2e} | "
                  f"GN (pre/post): {gn_before:.3f}/{gn_after:.3f}")
        
        log_rows.append({
            "step": opt_step + 1, 
            "loss": accum_loss, 
            "lr": lr_now,
            "gn_before": gn_before,
            "gn_after": gn_after
        })

        if (opt_step + 1) % SAVE_EVERY == 0:
            path = STEP_CKPT.format(step=opt_step+1)
            torch.save(model.state_dict(), path)
            
        if (opt_step + 1) % 200 == 0:
            model.eval()
            print(f"  --- Sample at step {opt_step+1} ---")
            sample_ex = random.choice(dataset.data)
            prompt_str = f"<|im_start|>user\n{sample_ex['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
            inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            stopped = False
            with torch.no_grad():
                for _ in range(80):
                    sl, _ = model(inp_ids)
                    nxt = torch.argmax(sl[0, -1, :]).item()
                    inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                    if nxt == tok.eos_token_id: stopped = True; break
            raw = tok.decode(inp_ids[0], skip_special_tokens=False)
            gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").strip()
            import re
            gen_fn = (m.group(1) if (m := re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", gen)) else "NONE")
            print(f"  Expected: {sample_ex['function_name']} | Got: {gen_fn} | EOS: {stopped}")
            model.train()

        opt_step += 1

    step = opt_step  # for final reporting

    print(f"\n=== Stage 4B Training Complete ===")
    print(f"Steps     : {step}")
    print(f"First loss: {first_loss:.4f}" if first_loss else "First loss: N/A")
    print(f"Final loss: {(loss*GRAD_ACCUM).item():.4f}")
    print(f"Best loss : {best_loss:.4f}")
    print(f"NaN/Inf   : {nan_detected}")
    print(f"Best ckpt : {BEST_CKPT}")

    with open(RESULTS, "w") as f:
        json.dump({"steps": step, "first_loss": first_loss,
                   "best_loss": best_loss, "nan": nan_detected, "log": log_rows}, f, indent=2)

    # Quick sample
    model.eval()
    print("\n=== Sample Generations ===")
    data = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8")][:3]
    eos_id = tok.eos_token_id
    marker = "<|im_start|>assistant\n"
    for ex in data:
        prompt = f"<|im_start|>user\n{ex['prompt']}{tok.eos_token}\n{marker}"
        inp = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        stopped = False
        with torch.no_grad():
            for _ in range(160):
                logits, _ = model(inp)
                nxt = torch.argmax(logits[0,-1,:]).item()
                inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], 1)
                if nxt == eos_id: stopped = True; break
        raw = tok.decode(inp[0], skip_special_tokens=False)
        gen = raw.split(marker)[-1].replace(tok.eos_token,"").strip()
        import re
        gen_fn = (m.group(1) if (m := re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", gen)) else "NONE")
        print(f"\nExpected fn: {ex['function_name']} | Got fn: {gen_fn} | EOS: {stopped}")
        print(f"Output:\n{gen[:200]}")

if __name__ == "__main__":
    main()
