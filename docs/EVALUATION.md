# Reproducible Local Evaluation

Measured on 2026-09-03 with CPython 3.14 and NumPy 2.5.2. These figures cover the deterministic offline simulator running in-process without cluster overhead or network latency.

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
| **FD003 drift PSI (threshold 0.20)** | 6.9623 | Dual-fault degradation shift |
| **FD004 drift PSI (threshold 0.20)** | 9.9131 | Multi-condition + dual-fault shift |
| **SENSOR_BIAS drift PSI (threshold 0.20)** | 7.5054 | Sensor calibration offset (+3.5σ) |
| **SENSOR_DROPOUT drift PSI (threshold 0.20)** | 12.0885 | Severe channel attenuation |

---

## Model Performance & Adaptation

In the reference promotion scenario:
- **Baseline Production `v1` MAE on shifted domain**: `4.0924`
- **Candidate `v2` Validation MAE**: `0.7382` (**82.0% error reduction**)
- **Canary Shadow Window**: 100 shadow observations verified before automated promotion.
- **Rollback Proof**: When candidate `v2` was deliberately damaged with serving skew, the canary gate halted deployment, revoked candidate routing, and preserved production `v1`.

---

## Reproduction Commands

```bash
# 1. Run all 22 unit, integration, and web tests
make test

# 2. Run the 5-scenario detector matrix
make matrix

# 3. Trace automated promotion under FD002 drift
make demo

# 4. Trace automated rollback under injected regression
make rollback
```

Treat these as baseline engineering checks. CV claims should cite these measured numbers alongside the containerized Prometheus / Grafana profile recordings.
