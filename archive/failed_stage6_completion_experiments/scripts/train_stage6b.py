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

CKPT_DIR   = os.path.join(ROOT, "checkpoints")

STAGE6A_TRAIN_FILE = os.path.join(ROOT, "data", "stage6A_blueprint_train.jsonl")
STAGE3_TRAIN_FILE  = os.path.join(ROOT, "data", "stage3_paraphrase_train.jsonl")
STAGE5_TRAIN_FILE  = os.path.join(ROOT, "data", "stage5_teacher_distill.jsonl")

BASE_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage5_best.pt")
STEP_CKPT  = os.path.join(CKPT_DIR, "samat_next_350m_stage6a_step_{step}.pt")
FINAL_CKPT = os.path.join(CKPT_DIR, "samat_next_350m_stage6a_best.pt")

MAX_SEQ     = 512
BATCH       = 1
GRAD_ACCUM  = 16
TOTAL_STEPS = 300
WARMUP      = 100
BASE_LR     = 2e-7
GRAD_CLIP   = 0.5
SEED        = 42

class MixedStageDataset(Dataset):
    def __init__(self, tok, max_seq):
        self.tok = tok
        self.max_seq = max_seq
        self.eos_token = "<|im_end|>"
        
        self.examples = []
        
        # Load Stage 6A (Version A + Version B)
        s6 = [json.loads(l) for l in open(STAGE6A_TRAIN_FILE, encoding="utf-8")]
        skipped_6_a = 0
        skipped_6_b = 0
        for ex in s6:
            # Version A (Full)
            prompt_str_a = f"<|im_start|>user\nWrite a python function named {ex.get('function_name', 'func')} for {ex.get('task_type', 'task')}. Ensure it handles edge cases correctly.{self.eos_token}\n<|im_start|>assistant\n"
            target_str_a = ex["target_code_full"] + self.eos_token
            res_a = self.tokenize_and_mask(prompt_str_a, target_str_a)
            if res_a:
                res_a["source"] = "stage6_full"
                self.examples.append(res_a)
            else:
                skipped_6_a += 1
                
            # Version B (Completion)
            prompt_str_b = f"<|im_start|>user\nComplete the Python function below. Return only the indented function body. Do not repeat the function signature. Do not repeat the docstring. Do not use markdown.\n\n{ex['prompt']}\n<|im_end|>\n<|im_start|>assistant\n"
            target_str_b = ex["target_completion"] + self.eos_token
            res_b = self.tokenize_and_mask(prompt_str_b, target_str_b)
            if res_b:
                res_b["source"] = "stage6_comp"
                self.examples.append(res_b)
            else:
                skipped_6_b += 1
                
        # Load Stage 3 (Full-code replay)
        s3 = [json.loads(l) for l in open(STAGE3_TRAIN_FILE, encoding="utf-8")]
        skipped_3 = 0
        for ex in s3:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{self.eos_token}\n<|im_start|>assistant\n"
            target_str = ex["target"] + self.eos_token
            res = self.tokenize_and_mask(prompt_str, target_str)
            if res:
                res["source"] = "stage3"
                self.examples.append(res)
            else:
                skipped_3 += 1
                
        # Load Stage 5 (Full-code replay)
        s5 = [json.loads(l) for l in open(STAGE5_TRAIN_FILE, encoding="utf-8")]
        skipped_5 = 0
        for ex in s5:
            prompt_str = f"<|im_start|>user\n{ex['prompt']}{self.eos_token}\n<|im_start|>assistant\n"
            target_str = ex["teacher_target"] + self.eos_token
            res = self.tokenize_and_mask(prompt_str, target_str)
            if res:
                res["source"] = "stage5"
                self.examples.append(res)
            else:
                skipped_5 += 1
                
        print(f"Loaded: Stage 6A Full ({len(s6)-skipped_6_a}), Stage 6A Comp ({len(s6)-skipped_6_b}), Stage 3 ({len(s3)-skipped_3}), Stage 5 ({len(s5)-skipped_5})")
        
        # Now create the mixed dataset (40/40/10/10) dynamically when sampling
        self.s6_full_data = [e for e in self.examples if e.get("source") == "stage6_full"]
        self.s6_comp_data = [e for e in self.examples if e.get("source") == "stage6_comp"]
        self.s3_data = [e for e in self.examples if e.get("source") == "stage3"]
        self.s5_data = [e for e in self.examples if e.get("source") == "stage5"]
        
        # We will dynamically sample in __getitem__ to enforce 80/10/10 distribution
        # Fake length since it's dynamic, large enough to cover TOTAL_STEPS
        self.virtual_len = TOTAL_STEPS * BATCH * GRAD_ACCUM * 2 

    def tokenize_and_mask(self, prompt_str, target_str):
        full_str = prompt_str + target_str
        full_ids = self.tok(full_str, add_special_tokens=False, max_length=self.max_seq, truncation=True).input_ids
        prompt_ids = self.tok(prompt_str, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]
        
        valid_target_tokens = sum(1 for l in labels if l != -100)
        if valid_target_tokens == 0:
            return None
            
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "valid_tokens": valid_target_tokens,
            "source": "stage6"
        }

    def __len__(self): 
        return self.virtual_len

    def __getitem__(self, idx):
        r = random.random()
        if r < 0.5:
            return random.choice(self.s6_full_data)
        elif r < 0.7:
            return random.choice(self.s6_comp_data)
        elif r < 0.85:
            return random.choice(self.s3_data)
        else:
            return random.choice(self.s5_data)

def collate(batch):
    ids_list = [ex["input_ids"] for ex in batch]
    lbl_list = [ex["labels"] for ex in batch]
    vtok_list = [ex["valid_tokens"] for ex in batch]
    
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
    print(f"Stage 6A Full-Scale Training | device={device}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    print(f"Loading base checkpoint: {BASE_CKPT}")
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))

    dataset = MixedStageDataset(tok, MAX_SEQ)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                         collate_fn=collate, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_schedule(s, WARMUP, TOTAL_STEPS))

    opt_step = 0
    data_iter = iter(loader)
    
    print("\nSkipping BASELINE Evaluation since we already ran it...")
    # model.eval()
    # baseline_metrics = run_all_evals(model, tok, device)
    
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

        if opt_step % 50 == 0:
            path = STEP_CKPT.format(step=opt_step)
            torch.save(model.state_dict(), path)
            print(f"  >>> Checkpoint saved: {path}")
            
            print(f"\nRunning Evaluation at Step {opt_step}...")
            model.eval()
            metrics = run_all_evals(model, tok, device)
            
            # Rollback logic checks
            stage5_pass = metrics.get("Stage 5", {}).get("pass_rate", 0)
            stage3_pass = metrics.get("Stage 3 Paraphrase", {}).get("pass_rate", 0)
            eos_rate = metrics.get("Stage 6B Natural Holdout (Completion)", {}).get("eos_rate", 0)
            rep_rate = metrics.get("Stage 6B Natural Holdout (Completion)", {}).get("rep_rate", 0)
            syn_rate = metrics.get("Stage 6B Natural Holdout (Completion)", {}).get("syntax_rate", 0)
            timeout_rate = metrics.get("Stage 6B Natural Holdout (Completion)", {}).get("timeout_rate", 0)
            
            failed = False
            fail_reasons = []
            
            if stage5_pass < 0.90:
                failed = True; fail_reasons.append(f"Stage 5 collapsed to {stage5_pass:.1%}")
            if stage3_pass < 0.80:
                failed = True; fail_reasons.append(f"Stage 3 collapsed to {stage3_pass:.1%}")
            if eos_rate < 0.92:
                failed = True; fail_reasons.append(f"EOS collapse to {eos_rate:.1%}")
            if rep_rate > 0.10:
                failed = True; fail_reasons.append(f"Repetition spike to {rep_rate:.1%}")
            if syn_rate < 0.20:
                failed = True; fail_reasons.append(f"Syntax collapse to {syn_rate:.1%}")
            if timeout_rate > 0.10:
                failed = True; fail_reasons.append(f"Timeout spike to {timeout_rate:.1%}")
                
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
