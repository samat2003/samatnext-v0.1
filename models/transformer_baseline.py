import json
import os

class TransformerConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 151936)
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.num_layers = kwargs.pop("num_layers", 16)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 12)
        self.intermediate_size = kwargs.pop("intermediate_size", 2048)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 8192)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, "num_key_value_heads", self.n_heads)
        self.head_dim = self.hidden_size // self.n_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, seq_len, _ = x.shape
        
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(B, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])

        # transpose for attention
        xq = xq.transpose(1, 2) # (B, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # MQA / GQA replication if needed (here we assume n_kv_heads == n_heads for simplicity if matching 12)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            xk = torch.repeat_interleave(xk, repeats=repeats, dim=1)
            xv = torch.repeat_interleave(xv, repeats=repeats, dim=1)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv) # (B, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.attn(self.input_layernorm(x), freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class TransformerForCausalLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            config.max_position_embeddings, 
            getattr(config, 'rope_theta', 10000.0)
        ))

    def forward(self, input_ids):
        B, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        
        # Causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[:seq_len], mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, None
