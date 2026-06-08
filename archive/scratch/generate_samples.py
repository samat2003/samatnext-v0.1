import os, sys, json, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM

def generate_samples():
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage6a_step_150.pt")
    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model = SamatNextForCausalLM(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.cpu().eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    
    dataset_path = os.path.join(ROOT, "data", "stage6A_blueprint_natural_holdout.jsonl")
    print("\n=== 1 COMPLETION SAMPLE (Step 100) ===", flush=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f][:1]
        
    for i, ex in enumerate(data):
        prompt = "<|im_start|>user\nComplete the Python function below. Return only the indented function body. Do not repeat the function signature. Do not repeat the docstring. Do not use markdown.\n\n" + ex["prompt"] + "\n<|im_end|>\n<|im_start|>assistant\n"
        inp_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
        with torch.no_grad():
            for _ in range(100):
                sl, _ = model(inp_ids)
                nxt = torch.argmax(sl[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device="cpu")], dim=1)
                if nxt == tokenizer.eos_token_id:
                    break
        gen_text = tokenizer.decode(inp_ids[0], skip_special_tokens=False)
        gen_text = gen_text.split("<|im_start|>assistant\n")[-1].replace(tokenizer.eos_token, "").strip()
        print(f"\n[Sample {i+1}]")
        print(gen_text, flush=True)
        
    print("\n=== 1 FULL-FUNCTION SAMPLE (Step 100) ===", flush=True)
    for i, ex in enumerate(data[:1]):
        prompt = "<|im_start|>user\n" + ex["task_prompt"] + "\n<|im_end|>\n<|im_start|>assistant\n"
        inp_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
        with torch.no_grad():
            for _ in range(100):
                sl, _ = model(inp_ids)
                nxt = torch.argmax(sl[0, -1, :]).item()
                inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device="cpu")], dim=1)
                if nxt == tokenizer.eos_token_id:
                    break
        gen_text = tokenizer.decode(inp_ids[0], skip_special_tokens=False)
        gen_text = gen_text.split("<|im_start|>assistant\n")[-1].replace(tokenizer.eos_token, "").strip()
        print(f"\n[Sample {i+1}]")
        print(gen_text, flush=True)

if __name__ == "__main__":
    generate_samples()
