import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.samat_next.config import SamatNextConfig
from models.samat_next.model import SamatNextForCausalLM

# ─── Configuration ───────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "samat_next_v0_1.json")
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "finetune", "step_1500_final.pt")

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

def load_model():
    """Load the fine-tuned SamatNext model."""
    print("=" * 60)
    print("  SamatNext v0.1 — Loading...")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    
    # Load model config and build architecture
    config = SamatNextConfig.from_json(CONFIG_PATH)
    config.vocab_size = len(tokenizer)
    model = SamatNextForCausalLM(config)
    
    # Load fine-tuned weights
    print(f"Loading weights from: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    
    # Move to GPU and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e6:.1f}M parameters on {device}")
    print()
    
    return model, tokenizer, device

def top_p_filter(logits, top_p):
    """Nucleus (top-p) sampling filter."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift so the first token above the threshold is kept
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response token-by-token with streaming output."""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Track generated tokens for the response
    generated_tokens = []
    eos_token_id = tokenizer.eos_token_id
    
    # Get special token IDs for stopping
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
        
        # Take logits for the last token only
        next_token_logits = logits[:, -1, :].float()
        
        # Apply repetition penalty
        if generated_tokens:
            for token_id in set(generated_tokens):
                next_token_logits[0, token_id] /= REPETITION_PENALTY
        
        # Apply temperature
        if TEMPERATURE > 0:
            next_token_logits = next_token_logits / TEMPERATURE
        
        # Apply top-p filtering
        filtered_logits = top_p_filter(next_token_logits[0], TOP_P)
        
        # Sample from the distribution
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        
        next_token_id = next_token.item()
        
        # Check for stop conditions
        if next_token_id == eos_token_id:
            break
            
        # Check for <|im_end|> token
        generated_tokens.append(next_token_id)
        if len(generated_tokens) >= len(im_end_token):
            if generated_tokens[-len(im_end_token):] == im_end_token:
                # Remove the im_end tokens from output
                generated_tokens = generated_tokens[:-len(im_end_token)]
                break
        
        # Stream the token to terminal
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
        print(token_text, end="", flush=True)
        
        # Append to input for next forward pass
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Truncate if we exceed context window (keep last 2048 tokens)
        if input_ids.shape[1] > 2048:
            input_ids = input_ids[:, -2048:]
    
    print()  # Newline after generation
    return tokenizer.decode(generated_tokens, skip_special_tokens=False)

def format_chatml(user_message):
    """Format the user message into ChatML format that the model was trained on."""
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def main():
    model, tokenizer, device = load_model()
    
    print("=" * 60)
    print("  SamatNext v0.1 — Interactive Chat")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'clear' to reset conversation.")
    print("=" * 60)
    print()
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\033[96mYou:\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not user_input.strip():
            continue
            
        if user_input.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break
            
        if user_input.strip().lower() == "clear":
            conversation_history = ""
            print("\n[Conversation cleared]\n")
            continue
        
        # Build the prompt with ChatML formatting
        conversation_history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        print("\033[93mSamatNext:\033[0m ", end="", flush=True)
        
        response = generate(model, tokenizer, device, conversation_history)
        
        # Add the response to history for multi-turn conversations
        conversation_history += response + "<|im_end|>\n"
        
        print()

if __name__ == "__main__":
    main()
