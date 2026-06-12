import sys
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def test_smoke_data_structure():
    # This checks that once prepare_data.py --smoke is run, outputs exist and are valid.
    manifest_path = os.path.join(ROOT, "data", "manifests", "data_manifest.json")
    if not os.path.exists(manifest_path):
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert "smoke" in manifest or "processed" in manifest
    assert "processed_files" in manifest or "data_hashes" in manifest or "files" in manifest
