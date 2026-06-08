"""
Stage 2A Fixed Evaluation
- Uses real EOS <|im_end|> to stop generation
- Proper <|im_start|> chat format
- Counts junk <|...|> tokens in output
- Reports syntax rate, repetition rate, whether EOS appears in outputs
"""
import os
import sys
import json
import ast
import re
import torch
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_unseen import UNSEEN_PROMPTS, check_syntax


# ── Generation (stops at real EOS) ───────────────────────────────────────────
def generate_fixed(model, tok, prompt, device, max_new_tokens=128):
    model.eval()
    eos_id  = tok.eos_token_id          # 151645 <|im_end|>
    eos_tok = tok.eos_token             # "<|im_end|>"

    prompt_text = f"<|im_start|>user\n{prompt}{eos_tok}\n<|im_start|>assistant\n"
    input_ids = tok(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    stopped_at_eos = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_id = torch.argmax(logits[0, -1, :]).item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if next_id == eos_id:
                stopped_at_eos = True
                break

    raw = tok.decode(input_ids[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    response = raw.split(marker)[-1] if marker in raw else raw
    return response, stopped_at_eos   # keep raw so EOS is visible


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_repetition(text):
    tokens = text.split()
    if len(tokens) < 10:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top > len(tokens) * 0.30

def count_junk(text):
    return len(re.findall(r'<\|[^>]*\|>', text))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to evaluate (default: best then latest)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    assert tok.eos_token_id == 151645
    assert tok.pad_token_id == 151643

    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model  = SamatNextForCausalLM(config).to(device)

    # Checkpoint priority: explicit arg > best > latest
    ckpt_candidates = [
        args.ckpt,
        os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2a_fixed_best.pt"),
        os.path.join(ROOT, "checkpoints", "samat_next_350m_stage2a_fixed_latest.pt"),
    ]
    ckpt_path = next((c for c in ckpt_candidates if c and os.path.exists(c)), None)
    if ckpt_path is None:
        print("ERROR: No checkpoint found.")
        sys.exit(1)

    print(f"Checkpoint : {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Load training log
    log_file = os.path.join(ROOT, "results", "stage2a_fixed_log.json")
    start_loss = final_loss = nan_happened = "N/A"
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
        entries = log.get("log", [])
        if entries:
            start_loss   = entries[0]["loss"]
            final_loss   = entries[-1]["loss"]
        nan_happened = log.get("nan_happened", "N/A")

    # Evaluate
    n = len(UNSEEN_PROMPTS)
    syntax_count = rep_count = junk_total = eos_stop_count = 0
    nan_inf_found = False
    samples = []
    all_results = []

    print(f"\nEvaluating {n} unseen prompts...\n")
    for i, item in enumerate(UNSEEN_PROMPTS):
        instr = item["instruction"]
        response, stopped = generate_fixed(model, tok, instr, device, max_new_tokens=128)

        valid = check_syntax(response.replace(tok.eos_token, "").strip())
        rep   = check_repetition(response)
        junk  = count_junk(response.replace(tok.eos_token, ""))   # don't count the real EOS

        if valid:   syntax_count  += 1
        if rep:     rep_count     += 1
        if stopped: eos_stop_count += 1
        junk_total += junk

        # NaN check
        enc = tok(
            f"<|im_start|>user\n{instr}{tok.eos_token}\n<|im_start|>assistant\n",
            add_special_tokens=False, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            lgts, _ = model(enc.input_ids)
            if not torch.isfinite(lgts).all():
                nan_inf_found = True

        res = {
            "instruction": instr,
            "generated":   response,
            "syntax_valid": valid,
            "repetitive":   rep,
            "junk_tokens":  junk,
            "stopped_at_eos": stopped,
        }
        all_results.append(res)
        if i < 5:
            samples.append(res)

    summary = {
        "checkpoint": ckpt_path,
        "start_loss":  start_loss,
        "final_loss":  final_loss,
        "nan_in_training": nan_happened,
        "nan_inf_in_eval": nan_inf_found,
        "syntax_valid_rate":   syntax_count / n,
        "repetition_rate":     rep_count    / n,
        "eos_stop_rate":       eos_stop_count / n,
        "total_junk_tokens":   junk_total,
        "samples":     samples,
        "all_results": all_results,
    }

    out = os.path.join(ROOT, "results", "stage2a_fixed_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== STAGE 2A FIXED EVALUATION ===")
    print(f"Checkpoint          : {ckpt_path}")
    print(f"Starting Loss       : {start_loss}")
    print(f"Final Loss          : {final_loss}")
    print(f"NaN in training     : {nan_happened}")
    print(f"NaN/Inf in eval     : {nan_inf_found}")
    print(f"Syntax Valid        : {syntax_count}/{n}  ({syntax_count/n*100:.1f}%)")
    print(f"Repetition Rate     : {rep_count}/{n}  ({rep_count/n*100:.1f}%)")
    print(f"Stopped at <|im_end|>: {eos_stop_count}/{n}  ({eos_stop_count/n*100:.1f}%)")
    print(f"Total junk tokens   : {junk_total}")
    print()
    print("--- 5 Sample Generations ---")
    for j, s in enumerate(samples, 1):
        print(f"\n[{j}] {s['instruction']}")
        print(f"    syntax={s['syntax_valid']}  rep={s['repetitive']}  "
              f"junk={s['junk_tokens']}  eos_stop={s['stopped_at_eos']}")
        print(f"    Output:\n{s['generated']}")


if __name__ == "__main__":
    main()
