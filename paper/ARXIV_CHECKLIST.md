# arXiv Submission Checklist — SamatNext v0.2-B

This checklist compiles validation checks to perform prior to submitting the source package of SamatNext v0.2-B to arXiv.

- [x] **Paper Compiles Cleanly**: Verified that the LaTeX source compiles from a clean directory using standard tools.
- [x] **No Unresolved Citations**: Checked compilation logs to ensure all citations resolve against `references.bib` or the compiled `.bbl` file.
- [x] **Table Value Consistency**: Checked that all tables in `main.tex`, `README.md`, `release_notes.txt`, and markdown results match the frozen JSON schemas under `results/tables/`.
- [x] **Neutral Claim Tone**: Verified that all positive hype phrases (e.g., "Transformer killer", "SOTA", "superior memory", "solves catastrophic forgetting") have been removed or placed in strictly negative/disclaimer contexts.
- [x] **Prompt Templates Documented**: Verified that the exact ChatML wrappers and HumanEval subset instruction formatting are documented in the paper and README.
- [x] **Parameter Count Recomputation**: Re-ran the verification scripts to prove that the hybrid model parameter count is exactly 356,083,208 (within 776 parameters or 0.001% of the Transformer baseline).
- [x] **Release Artifact Link**: Verified the link to the external reproducibility release package is active and matches.
- [x] **Licenses & Security Warnings**: Confirmed that checkpoint CC BY-NC-SA 4.0 licenses, dataset licenses, Qwen Research teacher licenses, and subprocess execution safety disclaimers are present.
- [x] **Limitations & Threats Section**: Included threats to validity (narrow synthetic curriculum, Stage 5 template bias, small model scale, context limitations, and lack of replay comparison) in the paper text.
