.PHONY: test lint fig
PYTHONPATH=src

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	python3 -m ruff check src tests scripts || true

fig:
	PYTHONPATH=src python3 scripts/make_figures.py
