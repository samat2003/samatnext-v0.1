import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_suite import run_all_evals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Baseline Eval on {device}...")
    
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        
    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model = SamatNextForCausalLM(config).to(device)
    
    ckpt = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")
    print(f"Loading {ckpt}...")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    
    # 1. Run eval suite
    metrics = run_all_evals(model, tok, device)
    
    # 2. Print 10 Completion Samples
    import json
    dataset_path = os.path.join(ROOT, "data", "stage6A_blueprint_natural_holdout.jsonl")
    print("\n" + "="*50)
    print("10 COMPLETION SAMPLES (Step 0 / Stage 5 Checkpoint)")
    print("="*50)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f][:10]
        
    for i, ex in enumerate(data):
        prompt_str = f"<|im_start|>user\nComplete the Python function below. Return only the indented function body. Do not repeat the function signature. Do not repeat the docstring. Do not use markdown.\n\n{ex['prompt']}\n<|im_end|>\n<|im_start|>assistant\n"
        inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        stopped_im_end = False
        stopped_eot = False
        with torch.no_grad():
            for _ in range(192):
                sl, _ = model(inp_ids)
                nxt = torch.argmax(sl[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == 151645:
                    stopped_im_end = True
                    break
                elif nxt == tok.eos_token_id:
                    stopped_eot = True
                    break
                    
        raw = tok.decode(inp_ids[0], skip_special_tokens=False)
        gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").replace("<|im_end|>", "").strip()
        print(f"[Sample {i+1}] Stop Token: {'im_end' if stopped_im_end else ('eot' if stopped_eot else 'NONE')}")
        print(gen)
        print("-" * 30)

if __name__ == "__main__":
    main()
