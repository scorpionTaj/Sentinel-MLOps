# Sentinel MLOps

> Sentinel is a self-healing MLOps reference pipeline: PSI-based drift detection triggers automated retraining, offline validation, canary release, and evidence-driven promotion/rollback — deterministic end-to-end demo, FastAPI/Prometheus/Grafana observability profile, with a documented scale-out path (Kafka/Spark/Delta/Dagster/MLflow).

The real running stack executes locally in under a second with a pure **NumPy core loop**, **FastAPI serving**, and an optional **Prometheus + Grafana** Docker Compose observability profile. Distributed infrastructure tools (Kafka/Redpanda, Spark, Delta Lake, Dagster, MLflow) are explicitly documented scale-out seams in [`docs/DECISIONS.md`](docs/DECISIONS.md), not running background services.

## Prove the loop

```bash
make test
make demo       # drift → retrain → canary → promote
make rollback   # drift → retrain → canary → injected regression → rollback
make matrix     # detect FD002, FD003, and FD004 regimes
```

No downloaded dataset or running infrastructure is needed for these commands. Each demo prints the
batch events, registry aliases, model history, PSI score, and Prometheus metrics as JSON.

## Run the observable API

```bash
docker compose up --build
curl -X POST http://localhost:8000/simulate-drift
curl -X POST http://localhost:8000/simulate-drift
```

Open [http://localhost:8000](http://localhost:8000)—do not open `index.html` directly—for the
minimal control-room web app. It exposes
the same real drift, canary, promotion, and rollback endpoints used by the command-line demos. The
demo controls and their reset endpoint are intentionally bound to localhost by Compose.

If a default port is occupied, override it—for example,
`GRAFANA_PORT=3300 docker compose up --build`.

The first request crosses the PSI threshold and starts a canary. The second accumulates enough
evidence to promote it. To demonstrate rollback, reset the volume, start a canary, then inject a
candidate-only serving regression:

```bash
docker compose down -v
docker compose up --build -d
curl -X POST http://localhost:8000/simulate-drift
curl -X POST http://localhost:8000/simulate-regression
```

- API and OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana dashboard: [http://localhost:3000](http://localhost:3000)

## Interfaces

`HealingLoop` is the primary interface. `bootstrap(rows)` establishes the production model and PSI
reference; `process(rows)` owns all quality, detection, training, release, and evidence transitions;
`predict(features, request_id)` routes live traffic deterministically.

The current executable profile uses a rolling feature pipeline, strict data gate, custom PSI,
regularized linear RUL model, atomic file registry, deterministic weighted router, and dependency-free
Prometheus exporter.

**Running stack vs. planned scale-out path**:
- **What is running today**: In-process NumPy core loop, FastAPI HTTP service, file-backed model registry, and containerized Prometheus + Grafana observability.
- **Documented scale-out seams**: [Architecture decisions](docs/DECISIONS.md) detail the exact integration seams to swap in Kafka/Redpanda for event streaming, Spark + Delta Lake for medallion lakehouse storage, Dagster for asset orchestration, and MLflow for remote model registry management.

## Real NASA C-MAPSS data

Place any 26-column `train_FD001.txt` through `train_FD004.txt` file under `data/reference/`, then load it
with `sentinel.datasets.load_cmapps_training`. The project does not silently download or redistribute
the dataset. NASA currently publishes the dataset metadata and download resource through its
[Open Data portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data); availability can
change, so the offline generator remains the reproducible default.

See [the exact problem and drift strategy](docs/PROBLEM.md) and the
[reproducible local evaluation](docs/EVALUATION.md). The interactive architecture artifact is
generated from `architecture.json` and delivered as `architecture.html`.

## Demo evidence to record

1. Keep Grafana visible on the PSI, model version, and promotion/rollback panels.
2. Call `/simulate-drift`; show the canary alias in `/status`.
3. Call either `/simulate-drift` again (promotion) or `/simulate-regression` (rollback).
4. End on `/status` and the model-event history.

Target CV bullet after recording measured runs:

> Built a self-healing MLOps reference pipeline that detects feature drift with PSI, retrains and validates RUL models, canary-routes releases, and automatically promotes or rolls back from live shadow metrics (NumPy core loop, FastAPI serving, Prometheus/Grafana monitoring, with documented Kafka/Spark/Delta/Dagster/MLflow scale-out seams).
