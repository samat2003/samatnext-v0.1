import sys, json, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 16  # Moderate batch size to avoid OOM but maximize GPU

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_teacher_batch_fast.py <in_file> <out_file>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen2.5-Coder-3B-Instruct on {device} (Fast Batching mode)...")
    
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
        
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    data = [json.loads(l) for l in open(in_file)]
    
    # Resume logic
    processed_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for l in f:
                processed_ids.add(json.loads(l)["id"])
                
    data_to_process = [d for d in data if d["id"] not in processed_ids]
    
    print(f"Found {len(data)} total prompts. {len(processed_ids)} already processed. {len(data_to_process)} remaining.")
    
    out_f = open(out_file, "a")
    
    for i in range(0, len(data_to_process), BATCH_SIZE):
        batch_ex = data_to_process[i:i+BATCH_SIZE]
        
        texts = []
        for ex in batch_ex:
            prompt = "You are generating training data for a Python coding model.\nReturn only Python code.\nNo markdown.\nNo explanation.\nNo code fences.\nTask:\n" + ex["prompt"]
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Return ONLY valid Python code. No markdown formatting, no explanations, no backticks."},
                {"role": "user", "content": prompt}
            ]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
            
        for j, ex in enumerate(batch_ex):
            # Extract only the generated tokens
            gen_ids = outputs[j][len(inputs.input_ids[j]):]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            
            if gen_text.startswith("```python"): gen_text = gen_text[9:]
            if gen_text.startswith("```"): gen_text = gen_text[3:]
            if gen_text.endswith("```"): gen_text = gen_text[:-3]
            gen_text = gen_text.strip()
            
            ex["teacher_target"] = gen_text
            out_f.write(json.dumps(ex) + "\n")
            
        out_f.flush()
        
        if (i+BATCH_SIZE) % 100 < BATCH_SIZE: 
            print(f"  Done {min(i+BATCH_SIZE, len(data_to_process))}/{len(data_to_process)}")
            
    out_f.close()
    print("Generation complete.")

if __name__ == "__main__":
    main()
