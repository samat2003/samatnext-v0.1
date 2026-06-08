import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("Prompting...", flush=True)
texts = ["def foo():\n"] * 16
inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

print("Generating...", flush=True)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.8, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print("Done!", flush=True)
