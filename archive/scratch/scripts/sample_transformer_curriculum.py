import json, random, torch
from transformers import AutoTokenizer
from baseline.config import TransformerConfig
from baseline.model import TransformerForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
config = TransformerConfig(vocab_size=len(tok))
model = TransformerForCausalLM(config).cuda()
model.load_state_dict(torch.load("checkpoints/transformer_350m_baseline_curriculum_stage5_best.pt", weights_only=True))
model.eval()

with open("data/stage5_teacher_distill.jsonl") as f:
    data = [json.loads(l) for l in f.readlines()]
    
random.seed(42)
samples = random.sample(data, 20)

for i, s in enumerate(samples):
    prompt_str = f"<|im_start|>user\n{s['prompt']}{tok.eos_token}\n<|im_start|>assistant\n"
    inp_ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        for _ in range(128):
            sl, _ = model(inp_ids)
            nxt = torch.argmax(sl[0, -1, :]).item()
            inp_ids = torch.cat([inp_ids, torch.tensor([[nxt]], device="cuda")], dim=1)
            if nxt == tok.eos_token_id: break
    raw = tok.decode(inp_ids[0], skip_special_tokens=False)
    gen = raw.split("<|im_start|>assistant\n")[-1].replace(tok.eos_token, "").strip()
    print(f"Sample {i+1}: {gen}")
    print("-" * 40)
