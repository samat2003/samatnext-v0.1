# SPDX-License-Identifier: Apache-2.0
import os
import sys
import subprocess
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    subprocess.run(args, check=True)

def main():
    print("=== STARTING SMOKE REPRODUCTION PIPELINE ===")
    
    # 1. Prepare data smoke
    prepare_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "prepare_data.py"),
        "--smoke"
    ]
    run_cmd(prepare_cmd)
    
    # 2. Check contamination
    contamination_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "check_contamination.py")
    ]
    run_cmd(contamination_cmd)
    
    # 3. Train tiny SamatNext
    train_samat_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train.py"),
        "--model", "samatnext",
        "--model-config", os.path.join(ROOT, "configs", "samatnext_350m.json"),
        "--train", os.path.join(ROOT, "data", "processed", "train.jsonl"),
        "--val", os.path.join(ROOT, "data", "processed", "train.jsonl"), # use train as placeholder val for smoke speed
        "--output", os.path.join(ROOT, "results", "runs", "smoke_samatnext"),
        "--smoke",
        "--grad-accum", "1",
        "--save-every", "5",
        "--eval-every", "5"
    ]
    run_cmd(train_samat_cmd)
    
    # 4. Train tiny Transformer
    train_trans_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train.py"),
        "--model", "transformer",
        "--model-config", os.path.join(ROOT, "configs", "transformer_350m_matched.json"),
        "--train", os.path.join(ROOT, "data", "processed", "train.jsonl"),
        "--val", os.path.join(ROOT, "data", "processed", "train.jsonl"),
        "--output", os.path.join(ROOT, "results", "runs", "smoke_transformer"),
        "--smoke",
        "--grad-accum", "1",
        "--save-every", "5",
        "--eval-every", "5"
    ]
    run_cmd(train_trans_cmd)
    
    # 5. Run evaluation on HumanEval for both models
    eval_samat_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_generate.py"),
        "--model", "samatnext",
        "--model-config", os.path.join(ROOT, "configs", "samatnext_350m.json"),
        "--checkpoint", os.path.join(ROOT, "results", "runs", "smoke_samatnext", "latest_model.pt"),
        "--dataset", os.path.join(ROOT, "data", "benchmark", "humaneval.jsonl"),
        "--dataset-type", "humaneval",
        "--output-dir", os.path.join(ROOT, "results", "runs", "smoke_samatnext", "eval"),
        "--max-new-tokens", "10" # very fast generations
    ]
    run_cmd(eval_samat_cmd)
    
    eval_trans_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_generate.py"),
        "--model", "transformer",
        "--model-config", os.path.join(ROOT, "configs", "transformer_350m_matched.json"),
        "--checkpoint", os.path.join(ROOT, "results", "runs", "smoke_transformer", "latest_model.pt"),
        "--dataset", os.path.join(ROOT, "data", "benchmark", "humaneval.jsonl"),
        "--dataset-type", "humaneval",
        "--output-dir", os.path.join(ROOT, "results", "runs", "smoke_transformer", "eval"),
        "--max-new-tokens", "10"
    ]
    run_cmd(eval_trans_cmd)

    # 6. Load results and print smoke table
    samat_eval_manifest = os.path.join(ROOT, "results", "runs", "smoke_samatnext", "eval", "eval_manifest.json")
    trans_eval_manifest = os.path.join(ROOT, "results", "runs", "smoke_transformer", "eval", "eval_manifest.json")
    
    samat_rate = 0.0
    trans_rate = 0.0
    
    if os.path.exists(samat_eval_manifest):
        with open(samat_eval_manifest, "r") as f:
            samat_rate = json.load(f).get("pass_rate", 0.0)
            
    if os.path.exists(trans_eval_manifest):
        with open(trans_eval_manifest, "r") as f:
            trans_rate = json.load(f).get("pass_rate", 0.0)
            
    print("\n" + "="*50)
    print("=== SMOKE REPRODUCTION RESULTS ===")
    print("="*50)
    print(f"SamatNext (Smoke) Pass Rate  : {samat_rate:.1%}")
    print(f"Transformer (Smoke) Pass Rate: {trans_rate:.1%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
