PY ?= python

.PHONY: install test lint typecheck check demo bench swat smd msl smap psm clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

check: lint typecheck test

# End-to-end run on the built-in synthetic benchmark (no downloads needed).
demo:
	$(PY) -m gcad.cli run --config configs/synthetic.yaml

# Multi-dataset synthetic benchmark: 10 dataset seeds, 1 model seed.
bench:
	$(PY) -m gcad.cli bench-synthetic --config configs/synthetic.yaml --dataset-seeds 0 1 2 3 4 5 6 7 8 9

# Real benchmarks: place the data first, see docs/DATASETS.md. 10 seeds as in the paper.
swat smd msl smap psm:
	$(PY) -m gcad.cli run --config configs/$@.yaml

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache .coverage build src/*.egg-info
