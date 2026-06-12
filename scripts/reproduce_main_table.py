# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import argparse
import datetime
import hashlib
import shutil
import subprocess
import torch

# Try importing transformers safely
try:
    import transformers
except ImportError:
    transformers = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM
from models.transformer_baseline import TransformerConfig, TransformerForCausalLM
from scripts.eval_suite import evaluate_subset, generate_stage5_holdout

# Define model paths and configs
MODELS_CONFIG = {
    "Transformer Scratch -> Stage5": {
        "model_type": "transformer",
        "config": os.path.join(ROOT, "configs", "transformer_350m.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_stage5_best.pt"),
        "files": {
            "stage5": "eval_baseline_stage_5_teacher-style_holdout.json",
            "stage3": "eval_baseline_stage_3_paraphrase.json",
            "stage2e": "eval_baseline_stage_2e_adversarial.json"
        },
        "display_name": "Transformer",
        "training_path": "Scratch → Stage5"
    },
    "SamatNext Scratch -> Stage5": {
        "model_type": "samatnext",
        "config": os.path.join(ROOT, "configs", "samat_next_v0_1.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "samatnext_350m_scratch_stage5_best.pt"),
        "files": {
            "stage5": "eval_samatnext_scratch_stage_5_teacher-style_holdout.json",
            "stage3": "eval_samatnext_scratch_stage_3_paraphrase.json",
            "stage2e": "eval_samatnext_scratch_stage_2e_adversarial.json"
        },
        "display_name": "SamatNext",
        "training_path": "Scratch → Stage5"
    },
    "Transformer Curriculum lr=3e-6": {
        "model_type": "transformer",
        "config": os.path.join(ROOT, "configs", "transformer_350m.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_curriculum_stage5_best.pt"),
        "files": {
            "stage5": "eval_transformer_curriculum_stage_5_teacher-style_holdout.json",
            "stage3": "eval_transformer_curriculum_stage_3_paraphrase.json",
            "stage2e": "eval_transformer_curriculum_stage_2e_adversarial.json"
        },
        "display_name": "Transformer",
        "training_path": "Curriculum lr=3e-6"
    },
    "Transformer Curriculum Rescue lr=1e-5": {
        "model_type": "transformer",
        "config": os.path.join(ROOT, "configs", "transformer_350m.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_rescue_r1_best.pt"),
        "files": {
            "stage5": "eval_transformer_rescue_r1_stage_5_teacher-style_holdout.json",
            "stage3": "eval_transformer_rescue_r1_stage_3_paraphrase.json",
            "stage2e": "eval_transformer_rescue_r1_stage_2e_adversarial.json"
        },
        "display_name": "Transformer",
        "training_path": "Curriculum Rescue lr=1e-5"
    },
    "Transformer Curriculum Rescue lr=3e-5": {
        "model_type": "transformer",
        "config": os.path.join(ROOT, "configs", "transformer_350m.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "transformer_350m_baseline_rescue_r2_best.pt"),
        "files": {
            "stage5": "eval_transformer_rescue_r2_stage_5_teacher-style_holdout.json",
            "stage3": "eval_transformer_rescue_r2_stage_3_paraphrase.json",
            "stage2e": "eval_transformer_rescue_r2_stage_2e_adversarial.json"
        },
        "display_name": "Transformer",
        "training_path": "Curriculum Rescue lr=3e-5"
    },
    "SamatNext Curriculum lr=3e-6": {
        "model_type": "samatnext",
        "config": os.path.join(ROOT, "configs", "samat_next_v0_1.json"),
        "checkpoint": os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt"),
        "files": {
            "stage5": "eval_stage_5_teacher-style_holdout.json",
            "stage3": "eval_stage_3_paraphrase.json",
            "stage2e": "eval_stage_2e_adversarial.json"
        },
        "display_name": "SamatNext",
        "training_path": "Curriculum lr=3e-6"
    }
}

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return "UNKNOWN"
    sha = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

def load_data(stage_key):
    if stage_key == "stage5":
        return generate_stage5_holdout()
    elif stage_key == "stage3":
        path = os.path.join(ROOT, "data", "stage3_paraphrase_eval.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif stage_key == "stage2e":
        path = os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown stage: {stage_key}")

def get_git_info():
    commit = "UNKNOWN"
    is_dirty = False
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        diff = subprocess.run(["git", "diff", "--quiet"], capture_output=True, cwd=ROOT)
        is_dirty = (diff.returncode != 0)
    except Exception:
        pass
    return commit, is_dirty

def get_dependency_versions():
    deps = {}
    for mod_name in ["torch", "transformers", "numpy", "tokenizers", "accelerate"]:
        try:
            mod = __import__(mod_name)
            deps[mod_name] = getattr(mod, "__version__", "UNKNOWN")
        except ImportError:
            deps[mod_name] = "NOT_INSTALLED"
    return deps

def get_gpu_vram():
    if torch.cuda.is_available():
        try:
            return f"{torch.cuda.get_device_properties(0).total_memory // 1024**2} MB"
        except Exception:
            return "UNKNOWN"
    return "N/A"

def run_evaluation(device, tokenizer_name, output_dir, timeout_seconds):
    os.makedirs(output_dir, exist_ok=True)
    datasets_dir = os.path.join(output_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    print(f"Loading tokenizer {tokenizer_name}...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    tokenizer_hash = "N/A"
    if os.path.exists(tokenizer_name):
        tokenizer_hash = compute_sha256(tokenizer_name)
        
    git_commit, git_dirty = get_git_info()
    dependency_versions = get_dependency_versions()
    
    results = {}
    eval_manifest = {
        "timeout_seconds": timeout_seconds,
        "metrics": {}
    }
    
    run_manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "exact_command": sys.argv,
        "git_commit_hash": git_commit,
        "git_dirty_status": git_dirty,
        "timeout_seconds": timeout_seconds,
        "output_directory": output_dir,
        "system_info": {
            "device": str(device),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if (torch.cuda.is_available() and hasattr(torch.version, 'cuda')) else "N/A",
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
            "gpu_vram": get_gpu_vram(),
            "dependencies": dependency_versions
        },
        "decoding_config": {
            "max_new_tokens": 192,
            "temperature": 0.0,
            "do_sample": False,
            "dtype": "torch.float32"
        },
        "seed_info": {
            "stage5_random_seed": 99999
        },
        "model_artifacts": {}
    }
    
    copy_list = []
    
    for model_key, model_meta in MODELS_CONFIG.items():
        print(f"\nEvaluating Model: {model_key}...")
        checkpoint_path = model_meta["checkpoint"]
        config_path = model_meta["config"]
        model_type = model_meta["model_type"]
        
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_key} because checkpoint is missing at {checkpoint_path}")
            continue
            
        checkpoint_hash = compute_sha256(checkpoint_path)
        config_hash = compute_sha256(config_path)
        
        run_manifest["model_artifacts"][model_key] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_hash": checkpoint_hash,
            "config_path": config_path,
            "config_hash": config_hash,
            "tokenizer_path_or_name": tokenizer_name,
            "tokenizer_hash": tokenizer_hash,
            "stages": {}
        }
        
        print(f"Loading model config from {config_path}...")
        if model_type == "samatnext":
            config = SamatNextConfig.from_json(config_path)
            model = SamatNextForCausalLM(config).to(device)
        else:
            config = TransformerConfig.from_json(config_path)
            model = TransformerForCausalLM(config).to(device)
            
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        important_missing = [k for k in missing_keys if "freqs_cis" not in k]
        important_unexpected = [k for k in unexpected_keys if "freqs_cis" not in k]
        if important_missing:
            raise RuntimeError(f"Missing key(s) in state_dict: {important_missing}")
        if important_unexpected:
            raise RuntimeError(f"Unexpected key(s) in state_dict: {important_unexpected}")
        model.eval()
        
        results[model_key] = {}
        eval_manifest["metrics"][model_key] = {}
        
        for stage_key in ["stage5", "stage3", "stage2e"]:
            print(f"Evaluating {stage_key}...")
            data = load_data(stage_key)
            
            if stage_key == "stage5":
                dataset_path = os.path.join(datasets_dir, "stage5_holdout.jsonl")
                with open(dataset_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
            elif stage_key == "stage3":
                dataset_path = os.path.join(ROOT, "data", "stage3_paraphrase_eval.jsonl")
            elif stage_key == "stage2e":
                dataset_path = os.path.join(ROOT, "data", "stage2e_adversarial_holdout.jsonl")
                
            dataset_hash = compute_sha256(dataset_path)
            
            run_manifest["model_artifacts"][model_key]["stages"][stage_key] = {
                "eval_dataset_path": dataset_path,
                "eval_dataset_hash": dataset_hash
            }
            
            metrics, eval_results = evaluate_subset(
                f"{model_key}_{stage_key}", data, model, tokenizer, device, 
                return_details=True, timeout_seconds=timeout_seconds
            )
            
            results[model_key][stage_key] = metrics["pass_rate"]
            eval_manifest["metrics"][model_key][stage_key] = metrics
            
            detailed_records = []
            for item in eval_results:
                detailed_records.append({
                    "prompt": item["prompt"],
                    "reference": item["tests"],
                    "raw_generation": item["raw"],
                    "cleaned_generation": item["eval_gen"],
                    "pass_fail": "pass" if item["test_pass"] else "fail",
                    "syntax_status": "SUCCESS" if item["syntax_ok"] else "FAILED",
                    "execution_status": "SUCCESS" if item["test_pass"] else "FAILED",
                    "traceback_error_type": item["err_msg"] if item["err_msg"] else "None",
                    "timeout_status": item["err_msg"] == "Timeout",
                    "decoding_config": {
                        "max_new_tokens": 192,
                        "temperature": 0.0,
                        "do_sample": False
                    },
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_hash": checkpoint_hash,
                    "config_path": config_path,
                    "config_hash": config_hash,
                    "tokenizer_name_or_path": tokenizer_name,
                    "tokenizer_hash": tokenizer_hash,
                    "eval_dataset_path": dataset_path,
                    "eval_dataset_hash": dataset_hash
                })
                
            model_slug = model_key.lower().replace(' ', '_').replace('->', 'to')
            detail_filename = f"detail_{model_slug}_{stage_key}.json"
            detail_filepath = os.path.join(output_dir, detail_filename)
            with open(detail_filepath, "w", encoding="utf-8") as f:
                json.dump(detailed_records, f, indent=2)
                
            cached_filename = f"cached_{model_slug}_{stage_key}.json"
            cached_filepath = os.path.join(output_dir, cached_filename)
            
            summary_list = []
            for i, (ex, item) in enumerate(zip(data, eval_results)):
                summary_list.append({
                    "id": ex.get("id", f"{stage_key}_{i}"),
                    "prompt": ex.get("prompt"),
                    "target": ex.get("target", ex.get("tests", [""])[0] if ex.get("tests") else ""),
                    "gen": item["gen"],
                    "syntax_ok": item["syntax_ok"],
                    "stopped": item["stopped_im_end"] or item["stopped_eot"],
                    "test_pass": item["test_pass"],
                    "name_match": True,
                    "err_msg": item["err_msg"],
                    "exp_fn": ex.get("function_name", "NONE"),
                    "gen_fn": "NONE"
                })
                
            with open(cached_filepath, "w", encoding="utf-8") as f:
                json.dump(summary_list, f, indent=2)
                
            dest_cached_filename = model_meta["files"][stage_key]
            dest_cached_filepath = os.path.join(ROOT, "results", dest_cached_filename)
            copy_list.append((cached_filepath, dest_cached_filepath))
            
            print(f"  {stage_key} Pass Rate: {metrics['pass_rate']:.1%}")
            
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
    with open(os.path.join(output_dir, "eval_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=4)
        
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=4)
        
    for src, dest in copy_list:
        shutil.copyfile(src, dest)
        print(f"Updated cached scientific JSON at {dest}")
        
    return results

def load_cached_results():
    results = {}
    results_dir = os.path.join(ROOT, "results")
    
    for model_key, model_meta in MODELS_CONFIG.items():
        results[model_key] = {}
        for stage_key, filename in model_meta["files"].items():
            filepath = os.path.join(results_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing evaluation file: {filepath}. Run with --force-eval to generate them.")
            
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            total = len(data)
            passed = sum(1 for item in data if item.get("test_pass", False))
            rate = passed / total if total > 0 else 0.0
            results[model_key][stage_key] = rate
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-eval", action="store_true", help="Force run evaluations on model checkpoints instead of loading cached results.")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output", type=str, default=None, help="Output directory for fresh evaluation artifacts.")
    parser.add_argument("--timeout-seconds", type=float, default=2.0, help="Subprocess execution timeout limit for code tests.")
    args = parser.parse_args()
    
    is_cached = True
    
    if args.force_eval:
        is_cached = False
        if args.output is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = os.path.join(ROOT, "results", "runs", f"fresh_eval_{timestamp}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running full fresh evaluation on device: {device}")
        print(f"Saving all raw per-task output artifacts to: {args.output}")
        results = run_evaluation(device, args.tokenizer, args.output, args.timeout_seconds)
    else:
        print("Loading pre-computed evaluation results from results/...")
        try:
            results = load_cached_results()
        except FileNotFoundError as e:
            print(e)
            print("Falling back to running live evaluation...")
            is_cached = False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = os.path.join(ROOT, "results", "runs", f"fresh_eval_{timestamp}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results = run_evaluation(device, args.tokenizer, args.output, args.timeout_seconds)
            
    # Output JSON table format
    final_table = {}
    for model_key, model_meta in MODELS_CONFIG.items():
        if model_key in results and "stage5" in results[model_key]:
            final_table[model_key] = {
                "model_display_name": model_meta["display_name"],
                "training_path": model_meta["training_path"],
                "stage5_pass_rate": results[model_key]["stage5"],
                "stage3_retention_rate": results[model_key]["stage3"],
                "stage2e_pass_rate": results[model_key]["stage2e"]
            }
            
    tables_dir = os.path.join(ROOT, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    json_path = os.path.join(tables_dir, "main_retention_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_table, f, indent=4)
    print(f"Saved main retention table JSON report to {json_path}")
    
    # Generate Markdown table
    md_path = os.path.join(tables_dir, "main_retention_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SamatNext-v0.1: Curriculum Retention and Sequential Plasticity\n\n")
        f.write("| Model | Training Path | Stage 5 Pass | Stage 3 Retention | Stage 2E Pass |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        
        for model_key in [
            "Transformer Scratch -> Stage5",
            "SamatNext Scratch -> Stage5",
            "Transformer Curriculum lr=3e-6",
            "Transformer Curriculum Rescue lr=1e-5",
            "Transformer Curriculum Rescue lr=3e-5",
            "SamatNext Curriculum lr=3e-6"
        ]:
            if model_key not in final_table:
                continue
            row = final_table[model_key]
            
            m_name = f"**{row['model_display_name']}**"
            t_path = row['training_path']
            s5_rate = f"{row['stage5_pass_rate']:.1%}"
            s3_rate = f"{row['stage3_retention_rate']:.1%}"
            s2_rate = f"{row['stage2e_pass_rate']:.1%}"
            
            if "SamatNext" in m_name and "Curriculum" in t_path:
                s3_rate = f"**{s3_rate}**"
                s2_rate = f"**{s2_rate}**"
                
            f.write(f"| {m_name} | {t_path} | {s5_rate} | {s3_rate} | {s2_rate} |\n")
            
        # Detect if a fresh evaluation run has been performed to display the correct note
        fresh_note = None
        runs_dir = os.path.join(ROOT, "results", "runs")
        if os.path.exists(runs_dir):
            fresh_runs = [d for d in os.listdir(runs_dir) if d.startswith("fresh_eval_")]
            if fresh_runs:
                # Sort by directory name to get the latest run
                latest_run = sorted(fresh_runs)[-1]
                latest_run_dir = os.path.join(runs_dir, latest_run)
                manifest_path = os.path.join(latest_run_dir, "run_manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as mf:
                            manifest_data = json.load(mf)
                        ts = manifest_data.get("timestamp", "UNKNOWN")
                        if ts != "UNKNOWN":
                            try:
                                dt = datetime.datetime.fromisoformat(ts)
                                ts_friendly = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except Exception:
                                ts_friendly = ts
                        else:
                            ts_friendly = "UNKNOWN"
                        out_dir_friendly = f"results/runs/{latest_run}"
                        fresh_note = f"\n*Note: This table was generated from a fresh evaluation run on {ts_friendly}. Full per-example artifacts are saved in {out_dir_friendly}.*\n"
                    except Exception:
                        pass

        if not is_cached:
            out_formatted = args.output.replace("\\", "/")
            f.write(f"\n*Note: This table was generated from a fresh evaluation run on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Full per-example artifacts are saved in {out_formatted}.*\n")
        elif fresh_note:
            f.write(fresh_note)
        else:
            f.write("\n*Note: The current main table is generated from cached evaluation JSONs unless reproduce_main_table.py is run with --force-eval. Paper-grade reproduction requires --force-eval and full per-example eval artifacts.*\n")
            
    print(f"Saved main retention table Markdown report to {md_path}")
    
    # Automatically update README.md table programmatically
    readme_path = os.path.join(ROOT, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
            
        with open(md_path, "r", encoding="utf-8") as f:
            table_md = f.read()
            
        table_lines = table_md.split("\n")
        table_only_lines = [line for line in table_lines if line.strip().startswith("|") or line.strip().startswith("*Note:")]
        table_only_text = "\n".join(table_only_lines)
        
        import re
        pattern = r"(### Results Table\n\n).*?(\n\n## Parameter Matching)"
        table_only_text_escaped = table_only_text.replace("\\", "\\\\")
        new_content = re.sub(pattern, rf"\g<1>{table_only_text_escaped}\g<2>", readme_content, flags=re.DOTALL)
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Automatically updated README.md table programmatically.")

if __name__ == "__main__":
    main()
