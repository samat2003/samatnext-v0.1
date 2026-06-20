# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

def test_no_stale_commands_in_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stale_cmds = ["scripts/compare_models.py"]
    for cmd in stale_cmds:
        assert cmd not in readme, f"Stale command {cmd} found in README"

def test_no_hype_words_in_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    forbidden = [
        "86.8%",
        "Transformer killer",
        "solves catastrophic forgetting",
        "proves",
        "superior long-term memory",
        "fresh_eval_/"
    ]
    for word in forbidden:
        assert word not in readme, f"Forbidden phrase '{word}' found in README"

def test_fresh_values_present_in_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "100.0%" in readme
    assert "98.8%" in readme
    assert "12.0%" in readme

def test_qwen_license_wording_is_conservative():
    data_licenses = (ROOT / "DATA_LICENSES.md").read_text(encoding="utf-8")
    assert "Qwen Research license" in data_licenses
    assert "not Apache-2.0" in data_licenses or "Qwen2.5-Coder-3B model is subject to the Qwen Research license, not Apache-2.0" in data_licenses

def test_checkpoint_license_consistency():
    ckpt = (ROOT / "CHECKPOINT_LICENSE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    
    assert "CC BY-NC-SA 4.0" in ckpt
    assert "CC BY-NC-SA 4.0" in readme
    assert "CC BY-NC-SA 4.0" in model_card or "CHECKPOINT_LICENSE.md" in model_card

def test_safety_wording():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "execution sandboxes" not in readme
    assert "subprocess sandboxing" not in readme

def test_fresh_artifact_archive_wording():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "fresh_eval_<timestamp>" in readme
    assert "fresh_eval_/" not in readme
