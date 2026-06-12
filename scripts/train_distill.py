"""
Knowledge Distillation Script for SamatNext v0.1
Teacher: Qwen2.5-Coder-3B-Instruct (4-bit quantized, frozen)
Student: SamatNext 355M (trained from scratch)

Loss = alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student || teacher)

The student learns to mimic the teacher's full probability distribution
over the vocabulary, not just the one correct token. This transfers
the teacher's "understanding" of code structure much faster than
training on raw data alone.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

# ─── Distillation Hyperparameters ────────────────────────────────
TEMPERATURE = 3.0       # Softens probability distributions — higher = softer
ALPHA = 0.3             # Weight on hard label loss (0.3 = 30% CE, 70% KL)
LR = 2e-4               # Slightly higher LR — distillation signal is richer
TOTAL_STEPS = 3000      # ~2 epochs over Python CodeFeedback
WARMUP_STEPS = 200
GRAD_ACCUM = 16         # Smaller than before — two models take more memory
SEQ_LEN = 1024          # Shorter sequences — two forward passes per step
BATCH_SIZE = 1
SAVE_EVERY = 500

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def _init_weights(module):
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

def is_main():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0

class PythonChatDataset(torch.utils.data.Dataset):
    """CodeFeedback filtered to Python, formatted as ChatML."""
    def __init__(self, tokenizer, seq_len=1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []

        if is_main():
            print("Loading CodeFeedback dataset (Python-only)...")

        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")

        python_kws = ["python", "def ", "import ", "class ", "print(", ".py"]
        kept = 0
        for row in ds:
            q = row.get("query", "")
            a = row.get("answer", "")
            if not any(kw in (q + a).lower() for kw in python_kws):
                continue
            prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > seq_len + 1:
                tokens = tokens[:seq_len + 1]
            self.examples.append(tokens)
            kept += 1

        if is_main():
            print(f"  Loaded {kept} Python examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        pad_id = self.tokenizer.eos_token_id
        pad_len = (self.seq_len + 1) - len(tokens)
        if pad_len > 0:
            tokens = tokens + [pad_id] * pad_len
        input_ids = tokens[:-1]
        labels = tokens[1:]
        labels = [l if l != pad_id else -100 for l in labels]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Combined hard + soft distillation loss.
    
    Hard loss: standard cross-entropy against ground truth labels
    Soft loss: KL divergence between student and teacher distributions
               at a higher temperature (softer = more information transferred)
    """
    # Hard label loss (standard cross-entropy)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # Soft label loss (KL divergence at temperature T)
    # Only compute on non-padding positions
    mask = labels.view(-1) != -100
    if mask.sum() == 0:
        return hard_loss

    s_logits_flat = student_logits.view(-1, student_logits.size(-1))[mask]
    t_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))[mask]

    # Align vocab sizes — teacher and student may differ slightly
    min_vocab = min(s_logits_flat.size(-1), t_logits_flat.size(-1))
    s_logits_flat = s_logits_flat[:, :min_vocab]
    t_logits_flat = t_logits_flat[:, :min_vocab]

    # Apply temperature scaling
    s_soft = F.log_softmax(s_logits_flat / temperature, dim=-1)
    t_soft = F.softmax(t_logits_flat / temperature, dim=-1)

    # KL(student || teacher) — student learns to match teacher's distribution
    kl_loss = F.kl_div(s_soft, t_soft, reduction="batchmean")

    # Scale by T^2 as per Hinton et al. (2015)
    soft_loss = (temperature ** 2) * kl_loss

    return alpha * hard_loss + (1 - alpha) * soft_loss

def train():
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "samat_next_v0_1.json")
    STUDENT_CKPT = os.path.join(ROOT_DIR, "checkpoints", "finetune", "step_1500_final.pt")
    CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "distill")

    if is_main_process:
        os.makedirs(CKPT_DIR, exist_ok=True)
        print("=" * 60)
        print("  SamatNext Knowledge Distillation")
        print("  Teacher: Qwen2.5-Coder-3B (4-bit)")
        print("  Student: SamatNext 355M")
        print("=" * 60)

    # ── 1. Load Teacher (frozen, 4-bit quantized) ──────────────────
    if is_main_process:
        print("\nLoading teacher model (Qwen2.5-Coder-3B in 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        quantization_config=bnb_config,
        device_map=f"cuda:{local_rank}"
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False  # Teacher is completely frozen

    teacher_vram = torch.cuda.memory_allocated() / 1e9
    if is_main_process:
        print(f"  Teacher loaded. VRAM: {teacher_vram:.2f} GB")

    # ── 2. Load Student ────────────────────────────────────────────
    if is_main_process:
        print("\nLoading student model (SamatNext 355M)...")
    config = SamatNextConfig.from_json(CONFIG_PATH)
    config.vocab_size = len(tokenizer)

    student = SamatNextForCausalLM(config)
    student.apply(_init_weights)
    _apply_zero_init(student)

    # Load fine-tuned weights as starting point
    if is_main_process:
        print(f"  Loading student weights from {STUDENT_CKPT}")
    ckpt = torch.load(STUDENT_CKPT, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        student.load_state_dict(ckpt)
    del ckpt

    student = student.to(local_rank).to(torch.bfloat16)
    student.train()

    total_vram = torch.cuda.memory_allocated() / 1e9
    if is_main_process:
        print(f"  Student loaded. Total VRAM: {total_vram:.2f} GB")

    # ── 3. Optimizer & Scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
    )

    # ── 4. Dataset ─────────────────────────────────────────────────
    dataset = PythonChatDataset(tokenizer, seq_len=SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # ── 5. Distillation Loop ───────────────────────────────────────
    if is_main_process:
        print(f"\nStarting distillation for {TOTAL_STEPS} steps...")
        print(f"  Temperature: {TEMPERATURE}, Alpha: {ALPHA}")
        print(f"  LR: {LR}, Grad Accum: {GRAD_ACCUM}\n")

    global_step = 0
    step = 0
    running_loss = 0.0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(local_rank)
        labels = labels.to(local_rank)

        # ── Teacher forward (no gradients) ──
        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids)
            teacher_logits = teacher_out.logits.detach()

        # ── Student forward ──
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits, _ = student(input_ids)

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                temperature=TEMPERATURE, alpha=ALPHA
            )
            loss = loss / GRAD_ACCUM

        loss.backward()

        running_loss += loss.item()

        is_last_accum = ((step + 1) % GRAD_ACCUM == 0)
        if is_last_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if is_main_process:
                print(f"Step {global_step:4d} | Loss: {running_loss:.4f}", flush=True)
            running_loss = 0.0

            # Save checkpoint
            if is_main_process and global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{global_step}.pt")
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            if global_step >= TOTAL_STEPS:
                if is_main_process:
                    final_path = os.path.join(CKPT_DIR, "step_final.pt")
                    torch.save(student.state_dict(), final_path)
                    print(f"\nDistillation complete! Final weights: {final_path}")
                break

        step += 1

    cleanup_ddp()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        with open("DISTILL_CRASH.txt", "w") as f:
            traceback.print_exc(file=f)
        raise
