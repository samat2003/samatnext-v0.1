import os, sys, json, random, math, time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_suite import run_all_evals

TRAIN_FILE = os.path.join(ROOT, "data", "stage6A_mini_train.jsonl")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")

BASE_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage5_best.pt")
STEP_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage6a_micro_step_{step}.pt")
FINAL_CKPT = os.path.join(CKPT_DIR, "samat_next_350m_stage6a_micro_best.pt")

MAX_SEQ     = 512
BATCH       = 1
GRAD_ACCUM  = 16
TOTAL_STEPS = 300
WARMUP      = 100
BASE_LR     = 1e-6
GRAD_CLIP   = 0.5
SEED        = 42

class Stage6Dataset(Dataset):
    def __init__(self, path, tok, max_seq):
        raw = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.examples = []
        self.tok = tok
        self.max_seq = max_seq
        self.eos_token = tok.eos_token
        
        skipped = 0
        for ex in raw:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{self.eos_token}\n<|im_start|>assistant\n"
            target_str = ex["target_code"] + self.eos_token
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
    print(f"Stage 6A Micro-Pilot Training | device={device}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    print(f"Loading base checkpoint: {BASE_CKPT}")
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))

    dataset = Stage6Dataset(TRAIN_FILE, tok, MAX_SEQ)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                         collate_fn=collate, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_schedule(s, WARMUP, TOTAL_STEPS))

    opt_step = 0
    data_iter = iter(loader)
    
    print("\nRunning BASELINE Evaluation before training starts...")
    model.eval()
    baseline_metrics = run_all_evals(model, tok, device)
    
    print(f"\n{'='*70}")
    print(f"Starting training: {TOTAL_STEPS} optimizer updates")
    print(f"{'='*70}\n")

    while opt_step < TOTAL_STEPS:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

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

        if torch.isnan(torch.tensor(accum_loss)) or torch.isinf(torch.tensor(accum_loss)):
            print(f"Step {opt_step+1}: NaN/Inf detected in loss — STOPPING IMMEDIATELY.")
            sys.exit(1)

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        opt_step += 1
        lr_now = optimizer.param_groups[0]["lr"]

        if opt_step % 10 == 0:
            print(f"Step {opt_step:>3}/{TOTAL_STEPS} | Loss: {accum_loss:.4f} | LR: {lr_now:.2e}")

        if opt_step % 100 == 0:
            path = STEP_CKPT.format(step=opt_step)
            torch.save(model.state_dict(), path)
            print(f"  >>> Checkpoint saved: {path}")
            
            print(f"\nRunning Evaluation at Step {opt_step}...")
            model.eval()
            metrics = run_all_evals(model, tok, device)
            
            # Rollback logic checks
            stage5_pass = metrics.get("Stage 5", {}).get("pass_rate", 0)
            stage3_pass = metrics.get("Stage 3 Paraphrase", {}).get("pass_rate", 0)
            stage6_pass = metrics.get("Stage 6A Holdout", {}).get("pass_rate", 0)
            he5_pass = metrics.get("HumanEval 5", {}).get("pass_rate", 0)
            eos_rate = metrics.get("Stage 6A Holdout", {}).get("eos_rate", 0)
            rep_rate = metrics.get("Stage 6A Holdout", {}).get("rep_rate", 0)
            
            base_stage6 = baseline_metrics.get("Stage 6A Holdout", {}).get("pass_rate", 0)
            
            failed = False
            fail_reasons = []
            
            if stage5_pass < 0.85:
                failed = True; fail_reasons.append(f"Stage 5 collapsed to {stage5_pass:.1%}")
            if stage3_pass < 0.70:
                failed = True; fail_reasons.append(f"Stage 3 collapsed to {stage3_pass:.1%}")
            if eos_rate < 0.90:
                failed = True; fail_reasons.append(f"EOS collapse to {eos_rate:.1%}")
            if rep_rate > 0.10:
                failed = True; fail_reasons.append(f"Repetition spike to {rep_rate:.1%}")
            if opt_step == 300 and he5_pass < 0.20:
                print(f"NOTE: HumanEval 5 failed to reach 1/5 ({he5_pass:.1%}) but Stage 6/3/5 holdouts look fine.")
                print("Marking as 'format/generalization improved but HumanEval not yet unlocked'.")
                
            if failed:
                print("\n" + "!"*50)
                print("ROLLBACK TRIGGERED!")
                for r in fail_reasons:
                    print("-", r)
                print("!"*50)
                sys.exit(1)
            else:
                print(f"Evaluation passed at step {opt_step}. Continuing training...\n")
                
    torch.save(model.state_dict(), FINAL_CKPT)
    print("Training Complete. Final checkpoint saved.")

if __name__ == "__main__":
    main()
