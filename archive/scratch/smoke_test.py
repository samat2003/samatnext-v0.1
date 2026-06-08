import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from train.losses import compute_lm_loss, compute_verifier_loss

def main():
    print("Initializing Samat-Next-Coder 150M prototype...")
    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SamatNextForCausalLM(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    verifier_labels = torch.empty(batch_size, device=device).random_(2).float()
    
    print("\nRunning forward pass...")
    lm_logits, verifier_logits = model(input_ids)
    
    print("Computing losses...")
    lm_loss = compute_lm_loss(lm_logits, labels)
    verifier_loss = compute_verifier_loss(verifier_logits, verifier_labels)
    
    total_loss = lm_loss + verifier_loss
    print(f"LM Loss: {lm_loss.item():.4f}")
    print(f"Verifier Loss: {verifier_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    print("\nRunning backward pass...")
    total_loss.backward()
    print("Backward pass successful!")
    
    assert not torch.isnan(total_loss).any(), "Loss contains NaN!"
    assert not torch.isinf(total_loss).any(), "Loss contains Inf!"
    
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx % 2 != 0:
            alpha = layer.mixer.alpha
            assert alpha.grad is not None, f"Alpha gradient missing in layer {layer_idx}"
            assert alpha.item() == 0.0, f"Alpha should be 0.0, got {alpha.item()}"
    
    print("\nSmoke test PASSED!")

if __name__ == "__main__":
    main()
