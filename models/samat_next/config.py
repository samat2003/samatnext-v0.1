import json
from dataclasses import dataclass

@dataclass
class SamatNextConfig:
    vocab_size: int = 151936
    num_layers: int = 16
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 8192
    attention_ratio: float = 0.5
    deltanet_ratio: float = 0.5
    use_differential_attention: bool = True
    use_verifier_head: bool = False
    rms_norm_eps: float = 1e-6
    mixer_pattern: str = "alternating"
    use_rope: bool = True
    use_lsm_rmsnorm: bool = False
    lsm_output_scale: float = 1.0

    @classmethod
    def from_json(cls, json_path: str) -> "SamatNextConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
