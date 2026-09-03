.PHONY: test demo rollback matrix benchmark web-check api up down

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python -m sentinel demo --scenario promote

rollback:
	PYTHONPATH=src python -m sentinel demo --scenario rollback

matrix:
	PYTHONPATH=src python -m sentinel matrix

benchmark:
	PYTHONPATH=src python -m sentinel benchmark

web-check:
	python scripts/check_web.py --base-url $${SENTINEL_URL:-http://127.0.0.1:8000}

api:
	uvicorn sentinel.api:app --app-dir src --reload

up:
	docker compose up --build

down:
	docker compose down
