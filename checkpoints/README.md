# SamatNext Checkpoints

This directory contains the checkpoints for the SamatNext and Transformer Baseline models.

**Note:** Large `.pt` and `.pth` checkpoint binaries are excluded from this repository via `.gitignore` to prevent bloating the GitHub history. If you have Git LFS configured, you may track them. Otherwise, you must train them locally or download them from an external release.

## Checkpoint Manifest

### SamatNext-v0.1
- **Best Stage 5 Checkpoint:** `samat_next_350m_stage5_best.pt`
- **Config Used:** `configs/samat_next_v0_1.json`
- **Exact Parameter Count:** 356,083,208 (Approx 356M)
- **Trainable Parameters:** 356,083,208

### Transformer Baseline
- **Best Stage 5 Checkpoint:** `transformer_350m_baseline_stage5_best.pt`
- **Config Used:** `configs/transformer_350m.json`
- **Exact Parameter Count:** 346,228,992 (Approx 346M)
- **Trainable Parameters:** 346,228,992

## Reproducing Evaluation Locally
To run the evaluation suite locally, ensure that the `.pt` files matching the names above are placed directly in this `checkpoints/` directory, and run:

```bash
python scripts/compare_models.py
```
