"""
Tiny Overfit with Fully Fixed Tokenizer + Template
===================================================
Fixes:
  1. Real Qwen chat format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
  2. Labels: -100 for prompt tokens AND pad tokens
  3. eos = <|im_end|>  (id 151645) — real stopping signal
  4. pad = <|endoftext|> (id 151643)
  5. Vocab-size check before training
  6. Embeddings resized if tokenizer IDs exceed model vocab_size
  7. NaN stops training immediately
  8. Reports: special tokens, vocab check, loss curve, 3 generations
"""

import os
import sys
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

# ── 1. Tokenizer setup ────────────────────────────────────────────────────────
def load_and_verify_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    # Verify the real tokens
    assert tok.eos_token == "<|im_end|>",  f"Unexpected eos: {tok.eos_token}"
    assert tok.eos_token_id == 151645,     f"Unexpected eos_id: {tok.eos_token_id}"
    assert tok.pad_token == "<|endoftext|>", f"Unexpected pad: {tok.pad_token}"
    assert tok.pad_token_id == 151643,     f"Unexpected pad_id: {tok.pad_token_id}"

    print("=== Tokenizer Special Tokens ===")
    print(f"  eos_token  : {repr(tok.eos_token):<30}  id={tok.eos_token_id}")
    print(f"  pad_token  : {repr(tok.pad_token):<30}  id={tok.pad_token_id}")
    print(f"  bos_token  : {repr(tok.bos_token)}")
    print(f"  unk_token  : {repr(tok.unk_token)}")
    print(f"  vocab_size : {tok.vocab_size}")
    max_special_id = max(tok.all_special_ids)
    print(f"  max special ID : {max_special_id}")
    print("================================\n")
    return tok, max_special_id

# ── 2. Vocab-size check & optional embedding resize ───────────────────────────
def check_and_resize(model, tok, max_special_id):
    model_vocab = model.config.vocab_size
    needed = max_special_id + 1
    print(f"=== Vocab Check ===")
    print(f"  model vocab_size      : {model_vocab}")
    print(f"  max tokenizer ID      : {max_special_id}")
    print(f"  embeddings sufficient : {model_vocab >= needed}")

    if model_vocab < needed:
        old_emb = model.model.embed_tokens
        new_emb = nn.Embedding(needed, old_emb.embedding_dim)
        new_emb.weight.data[:model_vocab] = old_emb.weight.data
        nn.init.normal_(new_emb.weight.data[model_vocab:], std=0.02)
        model.model.embed_tokens = new_emb

        old_lm = model.lm_head
        new_lm = nn.Linear(old_lm.in_features, needed, bias=False)
        new_lm.weight.data[:model_vocab] = old_lm.weight.data
        nn.init.normal_(new_lm.weight.data[model_vocab:], std=0.02)
        model.lm_head = new_lm

        model.config.vocab_size = needed
        print(f"  [WARN] Embeddings resized to {needed}")
    else:
        print(f"  [OK] No resize needed")
    print("===================\n")

# ── 3. Dataset: 20 tiny examples ─────────────────────────────────────────────
EXAMPLES = [
    ("Write a Python function that adds two numbers.",
     "def add(a, b):\n    return a + b"),
    ("Write a Python function that subtracts two numbers.",
     "def subtract(a, b):\n    return a - b"),
    ("Write a Python function that multiplies two numbers.",
     "def multiply(a, b):\n    return a * b"),
    ("Write a Python function that checks if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python function that checks if a number is odd.",
     "def is_odd(n):\n    return n % 2 != 0"),
    ("Write a Python function that computes the factorial of a number.",
     "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"),
    ("Write a Python function that returns the nth Fibonacci number.",
     "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"),
    ("Write a Python function that reverses a string.",
     "def reverse_string(s):\n    return s[::-1]"),
    ("Write a Python function that counts vowels in a string.",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("Write a Python function that finds the maximum in a list.",
     "def find_max(lst):\n    return max(lst)"),
    ("Write a Python function that sums a list of numbers.",
     "def sum_list(lst):\n    return sum(lst)"),
    ("Write a Python function that removes duplicates from a list.",
     "def remove_duplicates(lst):\n    return list(set(lst))"),
    ("Write a Python function that checks if a string is a palindrome.",
     "def is_palindrome(s):\n    return s == s[::-1]"),
    ("Write a Python function that converts Celsius to Fahrenheit.",
     "def c_to_f(c):\n    return (c * 9/5) + 32"),
    ("Write a Python function that returns the length of a string.",
     "def str_len(s):\n    return len(s)"),
    ("Write a Python function that sorts a list in ascending order.",
     "def sort_list(lst):\n    return sorted(lst)"),
    ("Write a Python function that checks if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True"),
    ("Write a Python function that counts words in a string.",
     "def count_words(s):\n    return len(s.split())"),
    ("Write a Python function that returns the absolute value of a number.",
     "def absolute_value(n):\n    return abs(n)"),
    ("Write a Python function that squares a number.",
     "def square(n):\n    return n ** 2"),
]

def build_input(tok, prompt, target):
    """
    Build token ids and labels for one example.
    Format:
        <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n  ← prompt mask
        {target}<|im_end|>                                              ← train on this
    """
    eos = tok.eos_token   # <|im_end|>
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")

    prompt_text = f"<|im_start|>user\n{prompt}{eos}\n<|im_start|>assistant\n"
    target_text = f"{target}{eos}"

    p_ids = tok(prompt_text,  add_special_tokens=False).input_ids
    t_ids = tok(target_text,  add_special_tokens=False).input_ids

    input_ids = p_ids + t_ids
    labels    = [-100] * len(p_ids) + t_ids   # mask prompt, train on target

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long),
    )

def collate(batch, pad_id):
    max_len = max(x[0].size(0) for x in batch)
    inp = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lbl = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        inp[i, :ids.size(0)]  = ids
        lbl[i, :labs.size(0)] = labs
    return inp, lbl

# ── 4. Generation (stops at real EOS) ────────────────────────────────────────
def generate(model, tok, prompt, device, max_new_tokens=80):
    model.eval()
    eos_id  = tok.eos_token_id
    eos_tok = tok.eos_token
    im_end  = eos_tok  # same token

    prompt_text = f"<|im_start|>user\n{prompt}{eos_tok}\n<|im_start|>assistant\n"
    input_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_id = torch.argmax(logits[0, -1, :]).item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if next_id == eos_id:
                break

    raw = tok.decode(input_ids[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    if marker in raw:
        response = raw.split(marker)[-1]
    else:
        response = raw
    return response   # keep EOS visible so user can see it stops correctly

# ── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Device: {device} | AMP dtype: {dtype}\n")

    tok, max_special_id = load_and_verify_tokenizer()

    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    model  = SamatNextForCausalLM(config).to(device)

    check_and_resize(model, tok, max_special_id)

    # Fresh random init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Build dataset
    examples = [build_input(tok, p, t) for p, t in EXAMPLES]

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fct  = nn.CrossEntropyLoss(ignore_index=-100)

    STEPS     = 300
    BATCH     = 4
    LOG_EVERY = 50
    first_loss = None
    last_loss  = None

    print("=== Tiny Overfit Training (300 steps) ===\n")
    model.train()

    for step in range(1, STEPS + 1):
        batch = random.choices(examples, k=BATCH)
        inp, lbl = collate(batch, tok.pad_token_id)
        inp = inp.to(device)
        lbl = lbl.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(inp)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lbl[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step}: NaN/Inf loss! Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

        if step % LOG_EVERY == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")

    print(f"\nTraining done. Start loss={first_loss:.4f}  Final loss={last_loss:.4f}\n")

    # Save
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(ROOT, "checkpoints", "samat_next_350m_tokenfix_overfit.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Checkpoint saved: {ckpt}\n")

    # 3 generation samples — show raw output so EOS is visible
    TEST_PROMPTS = [
        "Write a Python function that adds two numbers.",
        "Write a Python function that checks if a number is even.",
        "Write a Python function that reverses a string.",
    ]
    print("=== 3 Generation Samples (showing EOS token) ===\n")
    for i, p in enumerate(TEST_PROMPTS, 1):
        out = generate(model, tok, p, device)
        print(f"[{i}] Prompt : {p}")
        print(f"    Output :\n{out}")
        print()

if __name__ == "__main__":
    main()
