import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SamatNextConfig

class DifferentialAttention(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.q_proj_diff = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj_diff = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # Project — then immediately cast to float32 for all attention math
        q      = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        k      = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        v      = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        q_diff = self.q_proj_diff(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        k_diff = self.k_proj_diff(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        
        # Attention logits — clamp to safe range before masking
        attn_scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)).clamp(min=-1e4, max=1e4)
        diff_scores = (torch.matmul(q_diff, k_diff.transpose(-2, -1)) / math.sqrt(self.head_dim)).clamp(min=-1e4, max=1e4)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device))
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        diff_scores = diff_scores.masked_fill(~causal_mask, float('-inf'))

        # Detect all-masked rows (every position is -inf).
        # With a causal mask this should never happen (position i always sees itself),
        # but guard defensively for padded or zero-length sequences.
        all_masked_attn = (~causal_mask).all(dim=-1)   # (seq_len,)  — broadcast over batch/heads
        all_masked_diff = all_masked_attn               # same mask

        # For all-masked rows: zero out scores so softmax returns uniform 0, not NaN
        attn_scores = attn_scores.masked_fill(all_masked_attn.unsqueeze(0).unsqueeze(0), 0.0)
        diff_scores = diff_scores.masked_fill(all_masked_diff.unsqueeze(0).unsqueeze(0), 0.0)

        attn_probs = F.softmax(attn_scores, dim=-1)
        diff_probs = F.softmax(diff_scores, dim=-1)

        # Zero out probabilities for fully-masked rows
        attn_probs = attn_probs.masked_fill(all_masked_attn.unsqueeze(0).unsqueeze(0), 0.0)
        diff_probs = diff_probs.masked_fill(all_masked_diff.unsqueeze(0).unsqueeze(0), 0.0)

        # Debug assertions (active only in training; remove for production)
        if self.training:
            assert not torch.isnan(attn_scores).any(), "NaN in attn_scores"
            assert not torch.isnan(attn_probs).any(),  "NaN in attn_probs"
            assert not torch.isnan(diff_probs).any(),  "NaN in diff_probs"
        
        base_out = torch.matmul(attn_probs, v)
        diff_out = torch.matmul(diff_probs, v)
        
        alpha_casted = self.alpha.float()
        final_out = base_out + alpha_casted * diff_out

        assert not torch.isnan(final_out).any(), "NaN in DiffAttn final_out"
        
        final_out = final_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Cast back to original dtype before returning
        return self.o_proj(final_out.to(orig_dtype))
