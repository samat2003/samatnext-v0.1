import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from train.losses import compute_lm_loss, compute_verifier_loss

# 1. Dataset Generation
def get_dataset():
    base_examples = [
        ("adds two numbers", "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a - b\n"),
        ("subtracts two numbers", "def subtract(a, b):\n    return a - b\n", "def subtract(a, b):\n    return a + b\n"),
        ("multiplies two numbers", "def multiply(a, b):\n    return a * b\n", "def multiply(a, b):\n    return a / b\n"),
        ("checks if a number is even", "def is_even(n):\n    return n % 2 == 0\n", "def is_even(n):\n    return n % 2 != 0\n"),
        ("checks if a number is odd", "def is_odd(n):\n    return n % 2 != 0\n", "def is_odd(n):\n    return n % 2 == 0\n"),
        ("calculates the factorial of a number", "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n", "def factorial(n):\n    return n * factorial(n-1)\n"),
        ("returns the nth fibonacci number", "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n", "def fib(n):\n    return fib(n-1) + fib(n-2)\n"),
        ("reverses a string", "def reverse_string(s):\n    return s[::-1]\n", "def reverse_string(s):\n    return s[1:]\n"),
        ("counts the vowels in a string", "def count_vowels(s):\n    return sum(1 for c in s if c.lower() in 'aeiou')\n", "def count_vowels(s):\n    return len(s)\n"),
        ("finds the maximum in a list", "def find_max(lst):\n    return max(lst)\n", "def find_max(lst):\n    return min(lst)\n"),
        ("sums a list of numbers", "def sum_list(lst):\n    return sum(lst)\n", "def sum_list(lst):\n    return len(lst)\n"),
        ("removes duplicates from a list", "def remove_duplicates(lst):\n    return list(set(lst))\n", "def remove_duplicates(lst):\n    return lst\n"),
        ("checks if a string is a palindrome", "def is_palindrome(s):\n    return s == s[::-1]\n", "def is_palindrome(s):\n    return s != s[::-1]\n"),
        ("converts Celsius to Fahrenheit", "def c_to_f(c):\n    return (c * 9/5) + 32\n", "def c_to_f(c):\n    return (c * 5/9) + 32\n"),
        ("calculates the area of a rectangle", "def area_rect(w, h):\n    return w * h\n", "def area_rect(w, h):\n    return w + h\n"),
        ("calculates the area of a circle", "def area_circle(r):\n    import math\n    return math.pi * r**2\n", "def area_circle(r):\n    return 3.14 * r\n"),
        ("finds the length of a string", "def str_len(s):\n    return len(s)\n", "def str_len(s):\n    return 0\n"),
        ("sorts a list in ascending order", "def sort_list(lst):\n    return sorted(lst)\n", "def sort_list(lst):\n    return lst[::-1]\n"),
        ("checks if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n", "def is_prime(n):\n    return n % 2 != 0\n"),
        ("counts the number of words in a string", "def count_words(s):\n    return len(s.split())\n", "def count_words(s):\n    return len(s)\n")
    ]
    
    # Repeat/augment to get 50 examples
    dataset = []
    for i in range(50):
        action, correct, wrong = base_examples[i % len(base_examples)]
        dataset.append({
            "instruction": f"Write a Python function that {action}.",
            "correct_solution": correct,
            "wrong_solution": wrong
        })
    return dataset

# 2. Simple Generate Function
def generate_text(model, tokenizer, prompt_text, device, max_new_tokens=50):
    model.eval()
    input_text = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            lm_logits, _ = model(input_ids)
            next_token_logits = lm_logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    print("Starting Tiny Overfit Training for Samat-Next-Coder 150M...")
    
    # Settings
    batch_size = 4
    learning_rate = 3e-4
    steps = 500
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model & Config
    config_path = os.path.join(ROOT, "configs", "samat_next_150m.json")
    config = SamatNextConfig.from_json(config_path)
    model = SamatNextForCausalLM(config).to(device)
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = get_dataset()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        
        # We will manually build batches
        # Pick a batch of examples
        import random
        batch_examples = random.choices(dataset, k=batch_size)
        
        # 1. LM Training (Predicting correct solution)
        lm_texts = []
        for ex in batch_examples:
            lm_texts.append(f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['correct_solution']}")
            
        lm_encodings = tokenizer(lm_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        lm_input_ids = lm_encodings.input_ids
        
        # Forward for LM
        lm_logits, _ = model(lm_input_ids)
        lm_loss = compute_lm_loss(lm_logits, lm_input_ids)
        
        # 2. Verifier Training
        # We need pairs of instruction + correct, instruction + wrong
        verifier_texts = []
        verifier_labels = []
        for ex in batch_examples:
            # Coin flip for correct or wrong
            if random.random() > 0.5:
                verifier_texts.append(f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['correct_solution']}")
                verifier_labels.append(1.0)
            else:
                verifier_texts.append(f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['wrong_solution']}")
                verifier_labels.append(0.0)
                
        verifier_encodings = tokenizer(verifier_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        verifier_input_ids = verifier_encodings.input_ids
        verifier_labels_tensor = torch.tensor(verifier_labels, device=device)
        
        # Forward for Verifier
        _, verifier_logits = model(verifier_input_ids)
        verifier_loss = compute_verifier_loss(verifier_logits, verifier_labels_tensor)
        
        # Total loss
        total_loss = lm_loss + 0.2 * verifier_loss
        
        # Safety checks before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"ERROR: NaN/Inf detected in loss at step {step}!")
            break
            
        total_loss.backward()
        
        # Safety checks on gradients
        nan_in_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"ERROR: NaN/Inf detected in gradients of {name} at step {step}!")
                    nan_in_grads = True
                    break
        if nan_in_grads:
            break
            
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:03d} | LM Loss: {lm_loss.item():.4f} | Verifier Loss: {verifier_loss.item():.4f} | Total Loss: {total_loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")

    print("\nTraining complete.")
    
    # Save checkpoint
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(ROOT, "checkpoints", "samat_next_150m_tiny_overfit.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    
    # Generation Test
    print("\nRunning Generation Tests...")
    prompts = [
        "Write a Python function that adds two numbers.",
        "Write a Python function that checks if a number is even.",
        "Write a Python function that reverses a string."
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        generated = generate_text(model, tokenizer, p, device)
        clean_text = generated.encode('utf-8', errors='ignore').decode('utf-8')
        print(f"Output:\n{clean_text}")

if __name__ == "__main__":
    main()
