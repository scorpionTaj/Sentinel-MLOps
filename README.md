# Sentinel MLOps

> Sentinel is a self-healing MLOps reference pipeline: PSI-based drift detection triggers automated retraining, offline validation, canary release, and evidence-driven promotion/rollback — deterministic end-to-end demo, FastAPI/Prometheus/Grafana observability profile, with a documented scale-out path (Kafka/Spark/Delta/Dagster/MLflow).

The real running stack executes locally in under a second with a pure **NumPy core loop**, **FastAPI serving**, and an optional **Prometheus + Grafana** Docker Compose observability profile. Distributed infrastructure tools (Kafka/Redpanda, Spark, Delta Lake, Dagster, MLflow) are explicitly documented scale-out seams in [`docs/DECISIONS.md`](docs/DECISIONS.md), not running background services.

## Prove the loop

```bash
make test
make demo       # drift → retrain → canary → promote
make rollback   # drift → retrain → canary → injected regression → rollback
make matrix     # detect 7 C-MAPSS and synthetic stress regimes
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
control-room web app. It polls `/status` every 4 seconds and shows the evidence behind every
decision: PSI per batch against the batch-size-aware alert line, per-feature PSI and mean shift,
the canary's paired confidence interval against the promotion margin, and a timeline of pipeline
events with their details. The demo runs entirely on synthetic telemetry (210-row fleet batches);
real NASA C-MAPSS results live in [EVALUATION.md](docs/EVALUATION.md). The demo controls and their
reset endpoint are intentionally bound to localhost by Compose.

The dashboard discovers its choices from `GET /datasets`. It includes FD002–FD004 plus sensor-bias,
sensor-dropout, noise-burst, and adversarial-inversion stress scenarios. After starting the stack,
`make web-check` validates the page and assets and `make api-smoke` exercises every injectable regime.

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

## Data science workflow

The data-science side is organized so every reported number is regenerated from code:

```text
experiments/*.json          experiment configs (features, model, hyperparameter grid, seeds)
src/sentinel/evaluation/    metrics, engine-grouped splits, C-MAPSS features, models,
                            experiment runner, drift/canary calibration, report renderer
reports/                    generated JSON results (committed; never edited by hand)
notebooks/                  01 EDA · 02 drift & canary calibration · 03 model comparison
docs/EVALUATION.md          generated from reports/ by `make report`
docs/DATA_CARD.md           dataset provenance, labels, quirks
docs/MODEL_CARD.md          serving model vs offline candidates, limits
data/reference/MANIFEST.json  SHA-256 of every C-MAPSS file
```

```bash
pip install -e ".[dev,notebooks]"
make data        # download NASA C-MAPSS FD001–FD004 and verify checksums
make eval        # run every experiment config on real data
make calibrate   # Monte Carlo false-alarm, power and canary operating characteristic
make report      # regenerate docs/EVALUATION.md (runs all of the above)
make notebooks   # re-execute the notebooks
```

Headline results on the official C-MAPSS test sets (RMSE, RUL capped at 125, last-cycle protocol):

| Model | FD001 | FD002 | FD003 | FD004 |
| :--- | ---: | ---: | ---: | ---: |
| Predict the training-set mean | 41.9 | 44.9 | 43.7 | 45.6 |
| Serving ridge, 9 features, uncapped target (before) | 32.0 | 39.1 | 54.6 | 60.0 |
| Serving ridge, capped target (now serving) | 22.2 | 29.0 | 21.9 | 37.1 |
| Offline features + RBF kernel ridge (best offline) | 14.5 | 14.2 | 14.5 | 16.0 |

See [EVALUATION.md](docs/EVALUATION.md) for confidence intervals, NASA scores, drift false-alarm
and detection rates, and the canary operating characteristic. The live loop can also load real
files directly with `sentinel.datasets.load_cmapps_training` / `load_cmapps_test`; research code
uses `load_cmapps_arrays` (all 26 columns). The project never redistributes the dataset; NASA
publishes it via its [Open Data portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

See [the exact problem and drift strategy](docs/PROBLEM.md). The interactive architecture artifact is
generated from `architecture.json` and delivered as `architecture.html`. The detailed executable
flow is captured separately in [`sentinel-dataflow.html`](sentinel-dataflow.html), generated from
the validated [`sentinel-dataflow.json`](sentinel-dataflow.json) specification.

## Demo evidence to record

1. Keep Grafana visible on the PSI, model version, and promotion/rollback panels.
2. Call `/simulate-drift`; show the canary alias in `/status`.
3. Call either `/simulate-drift` again (promotion) or `/simulate-regression` (rollback).
4. End on `/status` and the model-event history.

Target CV bullet after recording measured runs:

> Built a self-healing MLOps reference pipeline that detects feature drift with PSI, retrains and validates RUL models, canary-routes releases, and automatically promotes or rolls back from live shadow metrics (NumPy core loop, FastAPI serving, Prometheus/Grafana monitoring, with documented Kafka/Spark/Delta/Dagster/MLflow scale-out seams).
