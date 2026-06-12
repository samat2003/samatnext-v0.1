import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def test_samat_next_imports():
    try:
        from models.samat_next.config import SamatNextConfig
        from models.samat_next.model import SamatNextModel, SamatNextForCausalLM
        from models.samat_next.layers import SamatNextBlock
        from models.samat_next.differential_attention import DifferentialAttention
        assert True
    except Exception as e:
        assert False, f"Failed to import SamatNext modules: {e}"

def test_transformer_imports():
    try:
        from models.transformer_baseline import TransformerConfig, TransformerForCausalLM
        assert True
    except Exception as e:
        assert False, f"Failed to import Transformer baseline modules: {e}"

def test_script_imports():
    # Verify we can import helper modules/functions
    # (Since scripts have __main__ blocks guarded, importing them shouldn't run training)
    try:
        import scripts.verify_config_truth as verify_config_truth
        import scripts.verify_repo_clean as verify_repo_clean
        assert True
    except Exception as e:
        assert False, f"Failed to import active script modules: {e}"
