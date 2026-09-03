# Problem statement and drift strategy

Sentinel estimates remaining useful life (RUL) from multivariate turbofan telemetry. A model
trained under the single-condition FD001 regime is exposed to the multi-condition FD002 regime;
the operating-condition and sensor distributions shift, causing the production error to rise.
Population Stability Index (PSI) compares every validated batch with the FD001 reference. A
threshold breach starts a retrain → offline validation → canary → promote/rollback workflow. The
repository ships a deterministic C-MAPSS-shaped generator so the proof works offline and a loader
for the official 26-column NASA files when they are available.

The deliberate rollback scenario corrupts only the canary artifact after it passes offline
validation. This represents packaging or serving skew, and proves that live shadow evidence—not
only an offline metric—protects production.

## Execution stack and architecture scope

The reference implementation runs entirely in-process using NumPy for the core pipeline and
FastAPI for serving, with an optional Docker Compose profile for Prometheus and Grafana.
Distributed infrastructure tools (Kafka/Redpanda, Spark, Delta Lake, Dagster, and MLflow) are
not active runtime services in this repository; they are documented architectural extension seams
described in [`docs/DECISIONS.md`](DECISIONS.md).

