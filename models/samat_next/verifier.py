import torch
import torch.nn as nn
from .config import SamatNextConfig

class VerifierHead(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_token_hidden = hidden_states[:, -1, :]
        return self.proj(last_token_hidden).squeeze(-1)
