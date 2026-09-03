# Reproducible local evaluation

Measured on 2026-09-03 with CPython 3.14 and NumPy 2.5.2. These numbers cover the deterministic
offline simulator, not Docker startup, Kafka/Spark transit, or a production observation window.

| Measure | Result |
|---|---:|
| Complete closed-loop runs | 20 |
| Expected promote/rollback decisions | 20/20 (100%) |
| Mean loop runtime | 0.0277 s |
| p95 loop runtime | 0.0359 s |
| Maximum loop runtime | 0.0502 s |
| Injected quality failures blocked | 1/1 (100%) |
| FD002 drift PSI (threshold 0.20) | 12.0885 |
| FD003 drift PSI (threshold 0.20) | 6.9623 |
| FD004 drift PSI (threshold 0.20) | 9.9131 |

In the promotion scenario, the recent-domain validation MAE improved from 4.0924 for production v1
to 0.7382 for candidate v2 (82.0% lower), then v2 was promoted after 100 shadow observations. In the
rollback scenario the same candidate passed offline validation, its served artifact was deliberately
damaged, and the canary gate retained production v1.

Reproduce the 16 unit and integration checks with `make test`; reproduce the three-regime detector
matrix with `make matrix`, and the two release event traces with `make demo` and `make rollback`.
Treat these as baseline engineering checks. CV claims should use the end-to-end Docker/Grafana
recording numbers once that run is captured on the target machine.
