# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from .config import SamatNextConfig
from .layers import SamatNextBlock, RMSNorm
from .verifier import VerifierHead

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class SamatNextModel(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SamatNextBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if getattr(config, "use_rope", True):
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads, 
                config.max_position_embeddings, 
                getattr(config, 'rope_theta', 10000.0)
            ))
        else:
            self.freqs_cis = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seq_len] if self.freqs_cis is not None else None
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class SamatNextForCausalLM(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.model = SamatNextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.use_verifier_head:
            self.verifier_head = VerifierHead(config)
        else:
            self.verifier_head = None

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.model(input_ids)
        lm_logits = self.lm_head(hidden_states)
        
        verifier_logits = None
        if self.verifier_head is not None:
            verifier_logits = self.verifier_head(hidden_states)
            
        return lm_logits, verifier_logits
