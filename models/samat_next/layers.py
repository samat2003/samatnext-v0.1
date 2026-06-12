# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SamatNextConfig
from .linear_state_mixer import DeltaNetInspiredLinearStateMixer
from .differential_attention import DifferentialAttention

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class MLP(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class SamatNextBlock(nn.Module):
    def __init__(self, config: SamatNextConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Determine mixer type based on mixer_pattern configuration
        pattern = getattr(config, "mixer_pattern", "alternating")
        num_layers = config.num_layers
        
        if pattern == "all_diffattn":
            self.mixer = DifferentialAttention(config)
        elif pattern == "all_linear_state":
            self.mixer = DeltaNetInspiredLinearStateMixer(config)
        elif pattern == "diffattn_first":
            if layer_idx < num_layers // 2:
                self.mixer = DifferentialAttention(config)
            else:
                self.mixer = DeltaNetInspiredLinearStateMixer(config)
        elif pattern == "linear_state_first":
            if layer_idx < num_layers // 2:
                self.mixer = DeltaNetInspiredLinearStateMixer(config)
            else:
                self.mixer = DifferentialAttention(config)
        else: # "alternating"
            if layer_idx % 2 == 0:
                self.mixer = DeltaNetInspiredLinearStateMixer(config)
            else:
                self.mixer = DifferentialAttention(config)
            
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, freqs_cis=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if isinstance(self.mixer, DifferentialAttention):
            hidden_states = self.mixer(hidden_states, freqs_cis)
        else:
            hidden_states = self.mixer(hidden_states)
            
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
