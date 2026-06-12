# SPDX-License-Identifier: Apache-2.0
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SamatNextConfig

def apply_rotary_emb_da(xq, xk, freqs_cis):
    # Shape of xq: [batch_size, seq_len, num_heads, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis: [seq_len, head_dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class DifferentialAttention(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_rope = getattr(config, "use_rope", True)
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.q_proj_diff = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj_diff = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis=None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q_diff = self.q_proj_diff(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_diff = self.k_proj_diff(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if configured and freqs_cis is provided
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rotary_emb_da(q, k, freqs_cis[:seq_len])
            q_diff, k_diff = apply_rotary_emb_da(q_diff, k_diff, freqs_cis[:seq_len])
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q_diff = q_diff.transpose(1, 2)
        k_diff = k_diff.transpose(1, 2)
        
        base_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff_out = F.scaled_dot_product_attention(q_diff, k_diff, v, is_causal=True)
        
        final_out = base_out + self.alpha * diff_out
        
        final_out = final_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(final_out)
