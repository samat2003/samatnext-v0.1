import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_teacher_batch.py <in_file> <out_file>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen2.5-Coder-3B-Instruct on {device}...")
    
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", 
            device_map="auto", 
            torch_dtype=torch.float16,
            local_files_only=True
        )
    except:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct", 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
    data = [json.loads(l) for l in open(in_file)]
    results = []
    
    print(f"Processing {len(data)} prompts...")
    for i, ex in enumerate(data):
        prompt = "You are generating training data for a Python coding model.\nReturn only Python code.\nNo markdown.\nNo explanation.\nNo code fences.\nTask:\n" + ex["prompt"]
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Return ONLY valid Python code. No markdown formatting, no explanations, no backticks."},
            {"role": "user", "content": prompt}
        ]
        
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
            
        gen_ids = outputs[0][len(inputs.input_ids[0]):]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Clean up if model leaked markdown anyway
        if gen_text.startswith("```python"): gen_text = gen_text[9:]
        if gen_text.startswith("```"): gen_text = gen_text[3:]
        if gen_text.endswith("```"): gen_text = gen_text[:-3]
        gen_text = gen_text.strip()
        
        ex["teacher_target"] = gen_text
        results.append(ex)
        
        if (i+1) % 10 == 0: print(f"  Done {i+1}/{len(data)}")
        
    with open(out_file, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
        
    print("Generation complete.")

if __name__ == "__main__":
    main()
