import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def test_eval_scripts_for_salvage():
    # Inspect scripts/eval_generate.py or eval_suite.py for signs of prefix salvage or hand patching.
    eval_suite_path = os.path.join(ROOT, "scripts", "eval_suite.py")
    if os.path.exists(eval_suite_path):
        with open(eval_suite_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Check for banned substrings representing hand patches or salvage
            assert "prefix_salvage" not in content
            assert "hand_patch" not in content
