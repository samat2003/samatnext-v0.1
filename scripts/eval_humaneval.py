# SPDX-License-Identifier: Apache-2.0
import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["samatnext", "transformer"])
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    dataset_path = os.path.join(ROOT, "data", "benchmark", "humaneval.jsonl")
    
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_generate.py"),
        "--model", args.model,
        "--model-config", args.model-config,
        "--checkpoint", args.checkpoint,
        "--dataset", dataset_path,
        "--dataset-type", "humaneval",
        "--output-dir", args.output_dir,
        "--timeout-seconds", str(args.timeout-seconds)
    ]
    
    print(f"Running HumanEval evaluation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
