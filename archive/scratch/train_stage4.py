"""
Stage 4 Training — Arbitrary Function-Name Copy Binding
========================================================
- Resume from samat_next_350m_stage3_best.pt
- fp32, lr=3e-6, grad_clip=0.5, 2000 steps
- Checkpoints every 500 steps + best
- Stops immediately on NaN/Inf
"""
import os, sys, json, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

def load_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    assert tok.eos_token_id == 151645
    assert tok.pad_token_id == 151643
    return tok

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.tok = tokenizer
        self.items = []
        self.skipped = 0
        eos = tokenizer.eos_token
        for idx, rec in enumerate(data):
            p, t = rec["prompt"].strip(), rec["target"].strip()
            p_ids = tokenizer(f"<|im_start|>user\n{p}{eos}\n<|im_start|>assistant\n",
                              add_special_tokens=False).input_ids
            t_ids = tokenizer(f"{t}{eos}", add_special_tokens=False).input_ids
            inp = (p_ids + t_ids)[:max_len]
            lbl = ([-100] * len(p_ids) + t_ids)[:max_len]
            if sum(1 for l in lbl if l != -100) == 0:
                self.skipped += 1; continue
            self.items.append((torch.tensor(inp, dtype=torch.long),
                               torch.tensor(lbl, dtype=torch.long)))
        print(f"Dataset: {len(self.items):,} kept, {self.skipped} skipped (target truncated)")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_fn(batch, pad_id):
    ml = max(x[0].size(0) for x in batch)
    inp = torch.full((len(batch), ml), pad_id, dtype=torch.long)
    lbl = torch.full((len(batch), ml), -100,   dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        L = ids.size(0); inp[i, :L] = ids; lbl[i, :L] = labs
    return inp, lbl

def get_lr(step, warmup, base_lr):
    return base_lr * step / max(1, warmup) if step <= warmup else base_lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int, default=2000)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--start_step", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Precision: float32\nTarget steps: {args.steps}  |  Resume: {args.resume}\n")

    tok = load_tokenizer()

    data_path = os.path.join(ROOT, "data", "stage4_name_binding_train.jsonl")
    data = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    print(f"Loaded {len(data):,} examples from {data_path}")

    dataset = CodeDataset(data, tok, max_len=512)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                         collate_fn=lambda b: collate_fn(b, tok.pad_token_id))

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    else:
        print("WARNING: No checkpoint found, training from random init.")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    BASE_LR, WARMUP, GRAD_CLIP, GRAD_ACCUM = 1e-6, 300, 0.5, 32
    SAVE_EVERY, LOG_EVERY = 500, 10

    optimizer  = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01, betas=(0.9, 0.95))
    loss_fct   = nn.CrossEntropyLoss(ignore_index=-100)

    from samat_next.deltanet import GatedDeltaNet
    from samat_next.differential_attention import DifferentialAttention
    dn_max, da_max = [], []
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m.register_forward_hook(lambda mod, inp, out, l=dn_max: l.append(out.float().abs().max().item()))
        elif isinstance(m, DifferentialAttention):
            m.register_forward_hook(lambda mod, inp, out, l=da_max: l.append(out.float().abs().max().item()))

    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"),     exist_ok=True)

    best_loss = float("inf")
    best_ckpt = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4_best.pt")
    log_data, nan_happened = [], False
    step_count, running_loss, first_loss = args.start_step, 0.0, None

    model.train()
    print(f"=== Training for {args.steps} steps ===\n")
    data_iter = iter(loader)

    for batch_idx in range(args.steps * GRAD_ACCUM):
        try:    inp, lbl = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); inp, lbl = next(data_iter)

        inp, lbl = inp.to(device), lbl.to(device)
        if (lbl != -100).sum() == 0:
            print(f"Batch {batch_idx}: 0 valid labels, skipping."); continue

        cur_lr = get_lr(step_count + 1, WARMUP, BASE_LR)
        for pg in optimizer.param_groups: pg["lr"] = cur_lr

        dn_max.clear(); da_max.clear()
        logits, _ = model(inp)
        sl = logits[..., :-1, :].contiguous()
        tl = lbl[..., 1:].contiguous()
        loss = loss_fct(sl.view(-1, sl.size(-1)), tl.view(-1)) / GRAD_ACCUM
        step_dn = max(dn_max) if dn_max else float("nan")
        step_da = max(da_max) if da_max else float("nan")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nNaN/Inf loss at batch {batch_idx}! Stopping."); nan_happened = True; break

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            bad = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if bad:
                print(f"\nNaN/Inf gradient at step {step_count+1}! Stopping."); nan_happened = True
                optimizer.zero_grad(); break

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            gn = sum(p.grad.float().norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
            optimizer.step(); optimizer.zero_grad(); step_count += 1
            step_loss = loss.item() * GRAD_ACCUM
            running_loss += step_loss
            if first_loss is None: first_loss = step_loss

            if step_count % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"Step {step_count:5d} | Loss: {avg:.4f} | LR: {cur_lr:.2e} | GN: {gn:.3f} "
                      f"| DN: {step_dn:.3f} | DA: {step_da:.3f} | VRAM: {vram:.2f}GB")
                log_data.append({"step": step_count, "loss": avg, "lr": cur_lr, "grad_norm": gn})
                if avg < best_loss:
                    best_loss = avg; torch.save(model.state_dict(), best_ckpt)
                    print(f"         -> Best checkpoint updated (loss={best_loss:.4f})")
                running_loss = 0.0

            if step_count % SAVE_EVERY == 0:
                ckpt = os.path.join(ROOT, "checkpoints", f"samat_next_350m_stage4_step_{step_count}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"         -> Checkpoint saved: {ckpt}")

            if step_count >= args.steps: break

    final_ckpt = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage4_latest.pt")
    torch.save(model.state_dict(), final_ckpt)

    log_file = os.path.join(ROOT, "results", "stage4_log.json")
    existing = []
    if os.path.exists(log_file):
        try: existing = json.load(open(log_file)).get("log", [])
        except: pass
    with open(log_file, "w") as f:
        json.dump({"log": existing + log_data, "nan_happened": nan_happened,
                   "best_loss": best_loss, "steps_completed": step_count}, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Steps     : {step_count}")
    print(f"First loss: {first_loss:.4f}" if first_loss else "N/A")
    print(f"Final loss: {log_data[-1]['loss']:.4f}" if log_data else "N/A")
    print(f"Best loss : {best_loss:.4f}")
    print(f"NaN       : {nan_happened}")
    print(f"Best ckpt : {best_ckpt}")

    # 3 quick sample generations
    TEST_PROMPTS = [
        ("standardize_str", "Create a Python function named standardize_str(text) that returns text stripped and lowercased."),
        ("flatten_nested",  "I need a Python function flatten_nested(items) that flattens items by one level."),
        ("mavol_sum",       "Write a Python function mavol_sum(a, b) that returns the sum of a and b."),
    ]
    eos_id, eos_tok = tok.eos_token_id, tok.eos_token
    print("\n=== 3 Sample Generations ===")
    model.eval()
    for exp_fn, prompt in TEST_PROMPTS:
        inp_ids = tok(f"<|im_start|>user\n{prompt}{eos_tok}\n<|im_start|>assistant\n",
                      add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        stopped = False
        with torch.no_grad():
            for _ in range(160):
                logits, _ = model(inp_ids)
                nxt = torch.argmax(logits[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == eos_id: stopped = True; break
        raw = tok.decode(inp_ids[0], skip_special_tokens=False)
        marker = "<|im_start|>assistant\n"
        resp = raw.split(marker)[-1] if marker in raw else raw
        import re
        gen_fn = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", resp)
        gen_fn = gen_fn.group(1) if gen_fn else "???"
        print(f"\nExpected fn: {exp_fn} | Got fn: {gen_fn} | EOS: {stopped}")
        print(f"Output:\n{resp}")

if __name__ == "__main__":
    main()
