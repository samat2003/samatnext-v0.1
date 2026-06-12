# SPDX-License-Identifier: Apache-2.0
# Makefile for SamatNext-v0.1 reproducibility pipeline

.PHONY: setup test prepare-data-smoke prepare-data bench-vram reproduce-smoke reproduce-main-table reproduce-main-table-fresh paper-check help

help:
	@echo "SamatNext-v0.1 Makefile targets:"
	@echo "  setup                 Install requirements"
	@echo "  test                  Run pytest test suite"
	@echo "  prepare-data-smoke    Build tiny smoke dataset, manifests, and hashes"
	@echo "  prepare-data          Build processed datasets and manifest"
	@echo "  bench-vram            Profile VRAM usage for SamatNext and Transformer"
	@echo "  reproduce-smoke       Run fast end-to-end smoke test"
	@echo "  reproduce-main-table  Generate tables from active checkpoints"
	@echo "  reproduce-main-table-fresh Run full fresh evaluation and output detail files"
	@echo "  paper-check           Verify paper draft, outline, checklists, and README tables"

setup:
	pip install -r requirements.txt

test:
	python -m pytest tests/

prepare-data-smoke:
	python scripts/prepare_data.py --smoke

prepare-data:
	python scripts/prepare_data.py

bench-vram:
	python scripts/bench_vram.py

reproduce-smoke:
	python scripts/reproduce_smoke.py

reproduce-main-table:
	python scripts/reproduce_main_table.py

reproduce-main-table-fresh:
	python scripts/reproduce_main_table.py --force-eval --timeout-seconds 5 --output results/runs/fresh_eval_$$(date +%Y%m%d_%H%M%S)

paper-check:
	python scripts/paper_check.py
