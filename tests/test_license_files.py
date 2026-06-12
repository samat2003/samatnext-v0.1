import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def test_license_files_exist():
    required_files = [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "DATA_LICENSES.md",
        "MODEL_CARD.md",
        "CHECKPOINT_LICENSE.md",
        "CITATION.cff",
        "SECURITY.md",
        "paper/draft_outline.md"
    ]
    
    # We will verify these files. During Step 1, these files might not exist yet,
    # but once we complete Step 2, they must exist and pass.
    # To make sure this test is useful, we'll write it to check them, and it will pass once created.
    missing = []
    for f in required_files:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            missing.append(f)
            
    # We raise AssertionError if any are missing once the setup is finished
    # For now, let's write it to assert they exist.
    assert len(missing) == 0, f"Missing required license/metadata files: {missing}"

def test_paper_license_section():
    outline_path = os.path.join(ROOT, "paper", "draft_outline.md")
    assert os.path.exists(outline_path), "Missing paper/draft_outline.md"
    
    with open(outline_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "Paper license decision" in content or "Paper License" in content
    assert "arXiv perpetual" in content or "CC BY 4.0" in content
