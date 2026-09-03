from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from sentinel.loop import HealingConfig, HealingLoop
from sentinel.scenarios import drift_scenario_ids
from sentinel.synthetic import generate_telemetry


def run_demo(state_dir: Path, scenario: str, reset: bool = False) -> dict[str, object]:
    if reset and state_dir.exists():
        shutil.rmtree(state_dir)
    loop = HealingLoop(
        state_dir,
        HealingConfig(
            drift_threshold=0.2,
            canary_weight=0.15,
            canary_min_observations=100,
            minimum_retrain_rows=40,
        ),
    )
    baseline = generate_telemetry("FD001", engines=8, cycles=45, seed=10)
    initial_version = loop.bootstrap(baseline)
    drift_batch = generate_telemetry("FD002", engines=2, cycles=35, seed=20, engine_offset=100)
    first = loop.process(drift_batch)
    if scenario == "rollback":
        loop.inject_canary_regression()
    followup_domain = "FD002"
    followup = generate_telemetry(followup_domain, engines=2, cycles=35, seed=30, engine_offset=200)
    second = loop.process(followup)
    return {
        "scenario": scenario,
        "initial_version": initial_version,
        "first_batch": first.to_dict(),
        "second_batch": second.to_dict(),
        "status": loop.status(),
        "metrics": loop.metrics.render(),
    }


def run_dataset_matrix() -> dict[str, object]:
    outcomes: dict[str, object] = {}
    for index, domain in enumerate(drift_scenario_ids(), start=1):
        with tempfile.TemporaryDirectory(prefix=f"sentinel-{domain.lower()}-") as temporary:
            loop = HealingLoop(
                Path(temporary),
                HealingConfig(drift_threshold=0.2, canary_min_observations=500),
            )
            loop.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
            result = loop.process(
                generate_telemetry(
                    domain, engines=3, cycles=35, seed=100 + index, engine_offset=index * 100
                )
            )
            outcomes[domain] = {
                "detected": result.drift.detected,
                "psi": result.drift.aggregate_psi,
                "event_kinds": [event.kind for event in result.events],
                "canary_version": result.canary_version,
            }
    return outcomes


def run_benchmark(reference_dir: Path | None = None) -> dict[str, object]:
    """Run end-to-end benchmark on large real NASA C-MAPSS dataset."""
    import time

    from sentinel.datasets import load_cmapps_training

    ref_dir = reference_dir or Path("data/reference")
    p1 = ref_dir / "train_FD001.txt"
    p2 = ref_dir / "train_FD002.txt"

    if not p1.exists() or not p2.exists():
        raise FileNotFoundError(
            f"NASA C-MAPSS dataset files not found under {ref_dir}. "
            "Please ensure train_FD001.txt and train_FD002.txt are present."
        )

    t0 = time.perf_counter()
    fd001_rows = load_cmapps_training(p1, "FD001")
    t_load1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    fd002_rows = load_cmapps_training(p2, "FD002")
    t_load2 = time.perf_counter() - t0

    with tempfile.TemporaryDirectory(prefix="sentinel-benchmark-") as temporary:
        loop = HealingLoop(
            Path(temporary),
            HealingConfig(drift_threshold=0.20, canary_min_observations=150),
        )
        t0 = time.perf_counter()
        initial_ver = loop.bootstrap(fd001_rows)
        t_boot = time.perf_counter() - t0

        t0 = time.perf_counter()
        drift_batch = fd002_rows[:500]
        drift_result = loop.process(drift_batch)
        t_proc1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        eval_batch = fd002_rows[500:1000]
        eval_result = loop.process(eval_batch)
        t_proc2 = time.perf_counter() - t0

        status = loop.status()

    return {
        "benchmark": "NASA_CMAPSS_LARGE_SCALE",
        "records_evaluated": len(fd001_rows) + len(fd002_rows),
        "fd001_rows": len(fd001_rows),
        "fd002_rows": len(fd002_rows),
        "load_time_seconds": round(t_load1 + t_load2, 4),
        "bootstrap_time_seconds": round(t_boot, 4),
        "batch_inference_and_drift_time_seconds": round(t_proc1, 4),
        "canary_evaluation_time_seconds": round(t_proc2, 4),
        "initial_production_version": initial_ver,
        "drift_psi": round(drift_result.drift.aggregate_psi, 4),
        "drift_detected": drift_result.drift.detected,
        "events": [e.kind for e in drift_result.events + eval_result.events],
        "final_production_version": status["aliases"].get("production"),
        "metrics": status["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sentinel self-healing MLOps demo")
    parser.add_argument("command", choices=["demo", "matrix", "benchmark"])
    parser.add_argument("--scenario", choices=["promote", "rollback"], default="promote")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.command == "matrix":
        result = run_dataset_matrix()
    elif args.command == "benchmark":
        result = run_benchmark()
    elif args.state_dir is None:
        with tempfile.TemporaryDirectory(prefix="sentinel-") as temporary:
            result = run_demo(Path(temporary), args.scenario)
    else:
        result = run_demo(args.state_dir, args.scenario, args.reset)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
