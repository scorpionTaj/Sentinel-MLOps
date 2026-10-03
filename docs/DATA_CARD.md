# Data Card: NASA C-MAPSS Turbofan Degradation

## Source and provenance

- **Dataset**: Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), Saxena et al.,
  PHM08. Simulated run-to-failure trajectories of turbofan engines. Metadata is published on the
  [NASA Open Data portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- **Retrieval**: `make data` downloads the 12 files from a public GitHub mirror (NASA does not
  serve stable per-file URLs) and fails unless every SHA-256 matches
  [`data/reference/MANIFEST.json`](../data/reference/MANIFEST.json). The raw `.txt` files are
  git-ignored and never redistributed.
- **Synthetic stand-in**: `sentinel.synthetic.generate_telemetry` produces C-MAPSS-shaped rows
  with 5 sensors for offline demos and tests. Synthetic numbers measure the control loop, not
  model quality.

## Contents

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
| :--- | ---: | ---: | :--- | :--- |
| FD001 | 100 | 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 (sea level) | HPC + fan degradation |
| FD004 | 249 | 248 | 6 | HPC + fan degradation |

Each row is one engine cycle with 26 columns: engine id, cycle, 3 operational settings
(altitude, Mach, throttle resolver angle) and 21 sensor channels. Exact row counts are in the
manifest.

## Labels

- **Train**: engines run to failure, so RUL at cycle *t* = last cycle − *t*.
- **Test**: trajectories stop before failure; `RUL_FD00x.txt` gives the remaining life after the
  last observed cycle, so RUL at *t* = last observed cycle + terminal RUL − *t*.
- **Piecewise-linear target**: early-life cycles are indistinguishable from healthy ones, so
  training and evaluation cap RUL at 125 cycles (`HealingConfig.rul_cap`, experiment `rul_cap`).

## Known characteristics that affect modelling

- **Constant channels**: s1, s5, s10, s16, s18 and s19 never move within an operating regime and
  s6 flickers between two readings in FD001/FD002. `OfflineFeatureBuilder` keeps a channel only
  if it takes at least 10 distinct values inside some regime, which leaves the 14 sensors used in
  the literature for FD001/FD002 (plus s6 for FD003/FD004, where the fan fault moves it).
- **Operating regimes**: FD002/FD004 sensors are dominated by six flight conditions. Without
  per-regime normalization the degradation signal is buried (see `notebooks/01_eda_cmapss.ipynb`).
- **Grouped, autocorrelated rows**: consecutive rows belong to the same engine. Splits must be by
  engine (`sentinel.evaluation.splits`) and drift batches should be fleet cross-sections; see
  [EVALUATION.md](EVALUATION.md) §2.
- **Engine ids restart at 1 in every file**, so any per-engine state is keyed by (domain, engine).

## Uses and limits

Suitable for benchmarking RUL estimation, covariate-shift detection between regimes, and release
gating. It is simulated data: absolute error levels do not transfer to real fleets, and the
regime shifts between subsets are far larger than typical in-service drift.
