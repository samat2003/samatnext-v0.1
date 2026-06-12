# SPDX-License-Identifier: Apache-2.0

.PHONY: setup test prepare-data-smoke prepare-data bench-vram reproduce-smoke reproduce-main-table reproduce-main-table-fresh paper-check help

help:
	@echo "SamatNext-v0.1 Reproducibility Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup                      - Install dependencies"
	@echo "  test                       - Run fast unit and hygiene tests"
	@echo "  prepare-data-smoke         - Download and tokenize smoke-test data (1%)"
	@echo "  prepare-data               - Download and tokenize full datasets"
	@echo "  bench-vram                 - Run GPU VRAM benchmark"
	@echo "  reproduce-smoke            - Run fast evaluation on smoke data"
	@echo "  reproduce-main-table       - Print cached artifact results for main table"
	@echo "  reproduce-main-table-fresh - Re-run evaluation models locally and output full artifacts"
	@echo "  paper-check                - Verify constraints, licenses, formatting, and formatting hygiene"

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests/

prepare-data-smoke:
	python scripts/prepare_data.py --smoke

prepare-data:
	python scripts/prepare_data.py

bench-vram:
	python scripts/benchmark_vram.py

reproduce-smoke:
	python scripts/reproduce_main_table.py --smoke

reproduce-main-table:
	python scripts/print_cached_table.py

reproduce-main-table-fresh:
	python scripts/reproduce_main_table.py --force-eval --output results/runs/fresh_eval_timestamp/

paper-check:
	python scripts/paper_check.py
