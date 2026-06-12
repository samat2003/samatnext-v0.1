# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

def test_file_line_counts():
    required_min_lines = {
        "README.md": 60,
        "DATA_LICENSES.md": 25,
        "CITATION.cff": 15,
        "scripts/paper_check.py": 80,
        "tests/test_claim_hygiene.py": 40,
        "tests/test_format_hygiene.py": 25,
        "results/tables/main_retention_table.md": 10,
        "Makefile": 20,
    }
    
    for rel_path, minimum in required_min_lines.items():
        path = ROOT / rel_path
        if path.exists():
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            assert line_count >= minimum, f"{rel_path} has only {line_count} lines, expected >= {minimum}"

def test_makefile_contents():
    makefile_path = ROOT / "Makefile"
    if makefile_path.exists():
        makefile = makefile_path.read_text(encoding="utf-8")
        assert "\nsetup:" in makefile
        assert "\nreproduce-main-table-fresh:" in makefile
        # Check that there are tab-indented recipes
        assert "\t" in makefile, "Makefile must contain tab-indented recipe lines"
