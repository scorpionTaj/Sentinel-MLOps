# Reproducible Local Evaluation

Updated on 2026-09-10 with CPython 3.14 and NumPy 2.5.x. These figures cover the deterministic offline simulator running in-process without cluster overhead or network latency.

---

## Benchmark Results

| Measure | Result | Notes |
| :--- | ---: | :--- |
| **Complete closed-loop runs** | 20 | Consecutive runs under automated harness |
| **Expected promote/rollback decisions** | 20/20 (100%) | Deterministic outcome verification |
| **Mean loop runtime** | 0.0277 s | Sub-50ms execution for entire retrain-eval loop |
| **p95 loop runtime** | 0.0359 s | Consistent latency profile |
| **Maximum loop runtime** | 0.0502 s | Cold-start ceiling |
| **Injected quality failures blocked** | 1/1 (100%) | NaN, null, and out-of-bounds rejection |
| **FD002 drift PSI (threshold 0.20)** | 12.0885 | Operating-condition shift |
| **FD003 drift PSI (threshold 0.20)** | 7.9528 | Dual-fault degradation shift |
| **FD004 drift PSI (threshold 0.20)** | 8.7198 | Multi-condition + dual-fault shift |
| **SENSOR_BIAS drift PSI (threshold 0.20)** | 8.5201 | Sensor calibration offset |
| **SENSOR_DROPOUT drift PSI (threshold 0.20)** | 12.0885 | Severe channel attenuation |
| **NOISE_BURST drift PSI (threshold 0.20)** | 3.2362 | High-variance acquisition window |
| **ADVERSARIAL drift PSI (threshold 0.20)** | 12.0885 | Inverted degradation relationship |

---

## Model Performance & Adaptation

In the reference promotion scenario:
- **Baseline Production `v1` MAE on shifted domain**: `4.0924`
- **Candidate `v2` Validation MAE**: `0.7382` (**82.0% error reduction**)
- **Canary Shadow Window**: 100 shadow observations verified before automated promotion.
- **Rollback Proof**: When candidate `v2` was deliberately damaged with serving skew, the canary gate halted deployment, revoked candidate routing, and preserved production `v1`.

---

## 3. Large-Scale NASA C-MAPSS Stress Test (74,390 Rows)

Beyond deterministic synthetic slices, Sentinel is benchmarked on the official, uncompressed **NASA C-MAPSS turbofan run-to-failure dataset**:

| Benchmark Parameter | Measured Value | Notes |
| :--- | ---: | :--- |
| **Total real records evaluated** | **74,390 rows** | Complete combined operational cycles |
| **Baseline training partition (`train_FD001.txt`)** | 20,631 rows | 100 aircraft turbofan engines run to failure |
| **Complex multi-condition shift (`train_FD002.txt`)** | 53,759 rows | 260 engines across 6 flight conditions (0–42 kft, 0–0.84 Mach) |
| **Test partition (`test_FD001.txt` + `RUL_FD001.txt`)** | 13,096 rows | Verified against true ground-truth RUL offsets |
| **Dataset parse & load time** | **0.380 s** | Ingested via `sentinel.datasets.load_cmapps_training` |
| **Baseline model fit (`v1` bootstrap)** | **0.185 s** | In-process NumPy ridge regression on 20,631 rows |
| **500-cycle batch drift scoring & canary deployment** | **0.050 s** | PSI calculation ($11.1349 > 0.20$) and automated candidate fit |
| **Shadow canary promotion evaluation** | **0.058 s** | Evaluated on real flight regimes and promoted to `v2` |
| **Peak memory overhead** | < 45 MB | Streamlined in-memory structures without Spark/JVM overhead |

---

## Reproduction Commands

```bash
# 1. Run all unit, integration, and real-scale dataset tests
make test

# 2. Run the 7-scenario detector matrix
make matrix

# Optional: with the API stack running, exercise every catalog regime
make api-smoke

# 3. Trace automated promotion under FD002 drift
make demo

# 4. Trace automated rollback under injected regression
make rollback

# 5. Run the 74,390-row large-scale NASA C-MAPSS benchmark
make benchmark
```

Treat these as baseline engineering checks. CV claims should cite these measured numbers alongside the containerized Prometheus / Grafana profile recordings.
