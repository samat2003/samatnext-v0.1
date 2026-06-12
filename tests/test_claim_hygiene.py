# SPDX-License-Identifier: Apache-2.0
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_readme_no_stale_commands():
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    stale_cmds = ["train_stage2a.py", "train_stage3.py", "train_stage5.py", "scripts/compare_models.py"]
    for cmd in stale_cmds:
        assert cmd not in readme, f"README contains stale training command: {cmd}"

def test_readme_no_stale_results():
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    # stale result numbers from previous runs
    assert "86.8%" not in readme, "README contains stale Stage 3 result (86.8%)"
    assert "97.6% Stage 5" not in readme, "README contains stale Stage 5 wording"
    # must contain fresh ones
    assert "83.0%" in readme, "README is missing fresh Stage 5 pass rate (83.0%)"
    assert "70.2%" in readme, "README is missing fresh Stage 3 retention rate (70.2%)"
    assert "4.3%" in readme, "README is missing fresh Stage 2E pass rate (4.3%)"

def test_data_licenses_qwen_claims():
    data_licenses_path = os.path.join(ROOT, "DATA_LICENSES.md")
    with open(data_licenses_path, "r", encoding="utf-8") as f:
        data_licenses = f.read()
    assert "Qwen Research license" in data_licenses, "DATA_LICENSES.md is missing 'Qwen Research license' citation"
    assert "Qwen2.5-Coder-3B model is subject to the Qwen Research license" in data_licenses
    assert "Apache License 2.0 (under Qwen2.5-Coder model terms)" not in data_licenses, "DATA_LICENSES.md wrongly claims Qwen2.5-Coder-3B outputs are Apache-2.0"

def test_checkpoint_license_consistency():
    readme_path = os.path.join(ROOT, "README.md")
    model_card_path = os.path.join(ROOT, "MODEL_CARD.md")
    ckpt_path = os.path.join(ROOT, "CHECKPOINT_LICENSE.md")
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    with open(model_card_path, "r", encoding="utf-8") as f:
        model_card = f.read()
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = f.read()
        
    assert "CC BY-NC-SA 4.0" in ckpt, "CHECKPOINT_LICENSE.md must state CC BY-NC-SA 4.0"
    assert "CC BY-NC-SA 4.0" in readme, "README must state CC BY-NC-SA 4.0"
    assert "CC BY-NC-SA 4.0" in model_card or "CHECKPOINT_LICENSE.md" in model_card, "MODEL_CARD.md must refer to checkpoint weights license"

def test_readme_no_unsafe_sandbox():
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read().lower()
        
    assert "execution sandboxes" not in readme, "README uses unsafe 'execution sandboxes' phrasing"
    assert "subprocess sandboxing" not in readme, "README uses unsafe 'subprocess sandboxing' phrasing"

def test_readme_gitignored_and_archive_pending():
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
        
    assert "gitignored" in readme, "README must mention that results/runs/ is gitignored"
    assert "External artifact archive: pending" in readme, "README must state 'External artifact archive: pending' unless real URL exists"
