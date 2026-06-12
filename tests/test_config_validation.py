import sys
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def test_samat_next_config_loading():
    from models.samat_next.config import SamatNextConfig
    
    # Try default config parameters
    cfg = SamatNextConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 768
    assert cfg.num_layers == 16

def test_transformer_config_loading():
    from models.transformer_baseline import TransformerConfig
    
    cfg = TransformerConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 768
    assert cfg.num_layers == 16
