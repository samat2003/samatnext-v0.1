import os, json, ast
from datasets import load_dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def is_clean_function(content):
    reject_keywords = [
        "HumanEval", "openai_humaneval", "canonical_solution", "check(candidate)", 
        "METADATA", "from humaneval", '"__main__"', "doctest.testmod", "unittest.main", 
        "```python", "eval(", "exec(", "subprocess", "os.system"
    ]
    if len(content) > 2000:
        return False, None, None, None
    for kw in reject_keywords:
        if kw in content:
            return False, None, None, None
            
    try:
        parsed_tree = ast.parse(content)
    except:
        return False, None, None, None
        
    funcs = 0
    func_node = None
    for node in parsed_tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return False, None, None, None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs += 1
            func_node = node
            
    if funcs != 1 or func_node is None:
        return False, None, None, None
        
    # Check if we can get a docstring
    docstring = ast.get_docstring(func_node)
    if not docstring or len(docstring.strip()) < 10:
        return False, None, None, None
        
    func_name = func_node.name
    return True, func_name, docstring, content

def build_dataset():
    print("Loading dataset in streaming mode...")
    ds = load_dataset("jon-tow/starcoderdata-python-edu", split="train", streaming=True)
    
    train_target = 5000
    holdout_target = 500
    
    train_data = []
    holdout_data = []
    
    pbar = tqdm(total=train_target + holdout_target, desc="Extracting Functions")
    
    for idx, ex in enumerate(ds):
        content = ex.get('content', '')
        repo_name = ex.get('max_stars_repo_name', 'unknown')
        score = ex.get('score', 0)
        
        ok, fname, doc, code = is_clean_function(content)
        if ok:
            # Clean up the docstring to use as prompt description
            desc = doc.strip().split('\n')[0] # just use the first line to be safe
            
            item = {
                "id": f"hf_python_edu_{idx}",
                "source": "jon-tow/starcoderdata-python-edu",
                "prompt": f"Write a Python function named {fname} that {desc}",
                "target_code": code.strip(),
                "function_name": fname,
                "task_family": "hf_python_edu",
                "validation_status": "accepted"
            }
            
            if len(train_data) < train_target:
                train_data.append(item)
                pbar.update(1)
            elif len(holdout_data) < holdout_target:
                holdout_data.append(item)
                pbar.update(1)
            else:
                break
                
    pbar.close()
    
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    
    train_path = os.path.join(ROOT, "data", "stage6c_hf_full_function_train.jsonl")
    holdout_path = os.path.join(ROOT, "data", "stage6c_hf_full_function_holdout.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(holdout_path, "w", encoding="utf-8") as f:
        for item in holdout_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(train_data)} train examples to {train_path}")
    print(f"Saved {len(holdout_data)} holdout examples to {holdout_path}")

if __name__ == "__main__":
    build_dataset()
