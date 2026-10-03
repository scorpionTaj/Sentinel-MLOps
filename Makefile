.PHONY: test lint demo rollback matrix benchmark data eval calibrate report notebooks web-check api-smoke api up down

PY := PYTHONPATH=src python
SENTINEL := $(PY) -m sentinel

test:
	$(PY) -m pytest

lint:
	ruff check src tests scripts

demo:
	$(SENTINEL) demo --scenario promote

rollback:
	$(SENTINEL) demo --scenario rollback

matrix:
	$(SENTINEL) matrix

# --- Data science workflow -------------------------------------------------------------------
# make data       download NASA C-MAPSS FD001-FD004 and verify SHA-256 against the manifest
# make eval       run every experiments/*.json config -> reports/experiments/*.json
# make calibrate  Monte Carlo drift + canary calibration -> reports/drift_calibration.json
# make report     regenerate docs/EVALUATION.md from reports/ (runs everything above)

data:
	python scripts/download_cmapps.py

benchmark: data
	$(SENTINEL) benchmark --out reports/benchmark.json

eval: data
	$(SENTINEL) evaluate

calibrate: data
	$(SENTINEL) calibrate --out reports/drift_calibration.json > /dev/null

report: eval calibrate benchmark
	$(SENTINEL) matrix --out reports/matrix.json > /dev/null
	$(SENTINEL) demo --scenario promote --out reports/demo_promote.json > /dev/null
	$(SENTINEL) demo --scenario rollback --out reports/demo_rollback.json > /dev/null
	$(SENTINEL) report

notebooks: data
	for nb in notebooks/*.ipynb; do \
		PYTHONPATH=$(CURDIR)/src jupyter nbconvert --to notebook --execute --inplace $$nb; \
	done

web-check:
	python scripts/check_web.py --base-url $${SENTINEL_URL:-http://127.0.0.1:8000}

api-smoke:
	python scripts/check_api_scenarios.py --base-url $${SENTINEL_URL:-http://127.0.0.1:8000}

api:
	uvicorn sentinel.api:app --app-dir src --reload

up:
	docker compose up --build

down:
	docker compose down
