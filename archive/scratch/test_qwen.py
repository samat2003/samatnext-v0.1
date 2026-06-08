import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", device_map="auto", torch_dtype=torch.float16, local_files_only=True)
tok.padding_side = "left"

req = """Create a new, unique Python coding task. Domain: strings. Difficulty: easy. Prompt format: HumanEval-style function signature + docstring.
Ensure the output code contains NO markdown formatting, NO prose, NO backticks, just the code itself.
CRITICAL: The solution MUST contain a docstring with at least 3 doctest examples (>>>).
Output exactly in this format:
[PROMPT]
The exact prompt text to give to the model.

[SOLUTION]
def the_function_name(...):
    \"\"\"
    >>> the_function_name(...)
    ...
    \"\"\"
    ...
"""

messages = [
    {"role": "system", "content": "You are a dataset generator for Python coding tasks. Always output the exact requested bracket format."},
    {"role": "user", "content": req}
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True)

gen_text = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("RAW GENERATION:")
print(gen_text)
