.PHONY: test lint fig cli-check smoke protocol verify manuscript audit-stats quickstart
PYTHONPATH=src
PY=PYTHONPATH=src python3

test:
	$(PY) -m pytest -q

lint:
	python3 -m ruff check src tests scripts examples || true

fig:
	$(PY) scripts/make_figures.py

cli-check: protocol verify

protocol:
	$(PY) -m glue_lp.cli protocol-check
	$(PY) scripts/check_protocol.py

verify:
	$(PY) -m glue_lp.cli verify-results
	$(PY) scripts/verify_results.py

manuscript:
	$(PY) scripts/build_manuscript.py

smoke:
	$(PY) scripts/smoke_synthetic.py

audit-stats:
	$(PY) scripts/apply_stats_from_json.py

quickstart:
	$(PY) examples/quickstart_synthetic.py
