import torch
import torch.nn as nn

def compute_lm_loss(lm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def compute_verifier_loss(verifier_logits: torch.Tensor, verifier_labels: torch.Tensor) -> torch.Tensor:
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(verifier_logits, verifier_labels)
    return loss
