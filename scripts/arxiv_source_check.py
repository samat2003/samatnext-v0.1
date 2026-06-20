# SPDX-License-Identifier: Apache-2.0
import os
import sys

def main():
    print("Checking arXiv source package...")
    if not os.path.exists('paper/main.tex'):
        print("FAIL: Missing paper/main.tex")
        sys.exit(1)
    if not (os.path.exists('paper/references.bib') or os.path.exists('paper/main.bbl')):
        print("FAIL: Missing bibliography files (references.bib or main.bbl)")
        sys.exit(1)
        
    accidental_files = []
    for r, d, files in os.walk('paper'):
        for f in files:
            if f.endswith('.pt') or f.endswith('.safetensors') or f.endswith('.tar.gz') or 'checkpoint' in r.lower() or 'runs' in r.lower():
                accidental_files.append(os.path.join(r, f))
                    
    if accidental_files:
        print("FAIL: Accidental files found in repository that shouldn't be in arXiv package:")
        for af in accidental_files[:10]:
            print(f"  - {af}")
        sys.exit(1)
        
    print("PASS: arXiv source package looks clean and ready!")
    sys.exit(0)

if __name__ == "__main__":
    main()
