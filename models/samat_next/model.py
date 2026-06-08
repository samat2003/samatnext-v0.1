import torch
import torch.nn as nn
from .config import SamatNextConfig
from .layers import SamatNextBlock, RMSNorm
from .verifier import VerifierHead

class SamatNextModel(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SamatNextBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
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
