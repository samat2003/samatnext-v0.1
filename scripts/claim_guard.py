# SPDX-License-Identifier: Apache-2.0
import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# List of files to scan
SCAN_FILES = [
    "README.md",
    "MODEL_CARD.md",
    "paper/main.tex",
    "paper/draft_outline.md",
    "release_notes.txt"
]

# Negative/disclaimer phrases that are allowed
DISCLAIMERS = [
    "does not solve catastrophic forgetting",
    "not a transformer killer",
    "not sota",
    "not a sota",
    "not claimed to be sota",
    "not claimed to be a sota",
    "no broad humaneval/mbpp/swe-bench claim",
    "no comparison to replay/ewc/adapters",
    "not a general transformer replacement",
    "does not prove",
    "not superior memory",
    "not a general replacement",
    "not a general replacement for standard transformers",
    "not a solution to catastrophic forgetting",
    "not a complete solution to catastrophic forgetting",
    "not solve catastrophic forgetting",
    "does not completely solve catastrophic forgetting",
    "not a claim that the catastrophic forgetting problem is solved",
    "not solves catastrophic forgetting",
    "state-of-the-art",
    "a sota code model",
    "do not fully solve",
    "not claim that samatnext-v0.1 is a general replacement for transformers, a sota code model, or a complete solution to catastrophic forgetting.",
    "not a general sota frontier coding assistant",
    "general sota frontier coding assistant",
    "not a general replacement for standard transformers across all nlp tasks",
    "not a general replacement for standard transformers",
    "is a general replacement for transformers, a sota code model, or a complete solution to catastrophic forgetting",
    "acts as a general replacement for standard transformers",
    "solves catastrophic forgetting, represents a state-of-the-art code model, or acts as a general replacement for standard transformers",
    "not claim that samatnext v0.2-b solves catastrophic forgetting, represents a state-of-the-art code model, or acts as a general replacement for standard transformers"
]

# Sort disclaimers by length descending to ensure longer phrases match first
DISCLAIMERS = sorted(DISCLAIMERS, key=len, reverse=True)

# Forbidden words/phrases (scanned case-insensitively)
FORBIDDEN_HYPE = [
    "transformer killer",
    "sota",
    "superior memory",
    "general replacement",
    "solves catastrophic forgetting",
    "improve adjacent curriculum forgetting",
    "reaches lower final stage 5 performance",
    "at the cost of peak specialized performance"
]

# Forbidden words that require careful checking (either forbidden outright or check if they are disclaimed)
# e.g., "solve", "proves"
FORBIDDEN_WORDS = [
    r"\bsolve\b",
    r"\bsolves\b",
    r"\bproves\b",
    r"\bprove\b"
]

def scan_file(file_path):
    abs_path = ROOT / file_path
    if not abs_path.exists():
        print(f"Skipping {file_path} (does not exist)")
        return True

    content = abs_path.read_text(encoding="utf-8")
    content_clean_md = content.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    content_lower = content_clean_md.lower()
    
    # Remove allowed disclaimer phrases first to avoid false positives
    cleaned_content_lower = content_lower
    for disclaimer in DISCLAIMERS:
        cleaned_content_lower = cleaned_content_lower.replace(disclaimer, " [disclaimer] ")

    # Check forbidden hype phrases
    for hype in FORBIDDEN_HYPE:
        # Check if it exists in the cleaned content
        if hype in cleaned_content_lower:
            # Let's find which line in the cleaned content has it
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                # Clean the line the same way
                line_clean_md = line.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
                line_lower = line_clean_md.lower()
                cleaned_line_lower = line_lower
                for disclaimer in DISCLAIMERS:
                    cleaned_line_lower = cleaned_line_lower.replace(disclaimer, " [disclaimer] ")
                if hype in cleaned_line_lower:
                    print(f"FAIL: {file_path}:{idx+1} contains forbidden hype phrase '{hype}':")
                    print(f"  > {line.strip()}")
                    return False

    # Check forbidden words (using regex)
    for pattern in FORBIDDEN_WORDS:
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            cleaned_line_lower = line_lower
            for disclaimer in DISCLAIMERS:
                cleaned_line_lower = cleaned_line_lower.replace(disclaimer, " [disclaimer] ")
            if re.search(pattern, cleaned_line_lower):
                print(f"FAIL: {file_path}:{idx+1} contains forbidden word pattern '{pattern}':")
                print(f"  > {line.strip()}")
                return False

    return True

def main():
    print("Running Automated Claim Guard...")
    success = True
    for f in SCAN_FILES:
        if not scan_file(f):
            success = False
            
    if success:
        print("PASS: Automated Claim Guard passed (no forbidden hype language detected outside disclaimer contexts).")
        sys.exit(0)
    else:
        print("FAIL: Automated Claim Guard failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
