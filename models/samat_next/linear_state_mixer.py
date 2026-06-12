# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from .config import SamatNextConfig

class DeltaNetInspiredLinearStateMixer(nn.Module):
    """
    DeltaNet-inspired linear-state mixer (simplified causal linear-state mixer).
    This is a simplified causal linear attention approximation, not a faithful 
    implementation of the full DeltaNet delta-rule update.
    """
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Non-linear activation for positive keys/queries
        q = torch.nn.functional.relu(q)
        k = torch.nn.functional.relu(k) + 1e-6

        # Causal linear attention via cumulative sum.
        # Cast to float32 for the cumsum + division to prevent bf16 overflow → NaN.
        orig_dtype = hidden_states.dtype
        q = q.float()
        k = k.float()
        v = v.float()

        kv = k * v
        kv_state = torch.cumsum(kv, dim=1)
        k_state  = torch.cumsum(k,  dim=1).clamp(min=1e-6)   # never zero

        out = (q * kv_state) / k_state
        out = out.to(orig_dtype)
        return self.o_proj(out)
