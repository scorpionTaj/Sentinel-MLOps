"""Render docs/EVALUATION.md from the JSON results under reports/ (never edit the output by hand)."""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_ORDER = (
    "mean_baseline",
    "serving_ridge_uncapped",
    "serving_ridge",
    "offline_ridge",
    "offline_quadratic_ridge",
    "offline_rff_ridge",
)
DOMAINS = ("FD001", "FD002", "FD003", "FD004")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    align = align or ":---" + "|---:" * (len(header) - 1)
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align.split("|")) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def _experiments_section(reports: Path) -> list[str]:
    results = {
        name: _load(reports / "experiments" / f"{name}.json") for name in EXPERIMENT_ORDER
    }
    results = {name: value for name, value in results.items() if value is not None}
    if not results:
        return ["_No experiment results found. Run `make data eval`._", ""]
    lines = [
        "Official protocol: each test engine is scored once at its last observed cycle against",
        "`RUL_FD00x.txt`, with the target capped at 125 cycles (the common C-MAPSS convention;",
        "uncapped-target RMSE is kept in the JSON). Hyperparameters are chosen by 5-fold",
        "engine-grouped cross-validation on the training file only; the test file is touched once.",
        "Values are mean ± std over 5 seeds (seeds only change randomized models).",
        "",
        "**Test RMSE (lower is better)**",
        "",
    ]
    rows = []
    for name, payload in results.items():
        row = [f"`{name}`"]
        for domain in DOMAINS:
            outcome = payload["results"].get(domain)
            if outcome is None:
                row.append("–")
                continue
            test = outcome["test"]["rmse"]
            spread = f" ± {test['std']:.2f}" if test["std"] > 0.005 else ""
            row.append(f"{test['mean']:.2f}{spread}")
        rows.append(row)
    lines += _table(["Experiment", *DOMAINS], rows)

    lines += ["**NASA PHM08 score (lower is better; penalizes late predictions)**", ""]
    rows = []
    for name, payload in results.items():
        rows.append(
            [f"`{name}`"]
            + [
                f"{payload['results'][d]['test']['nasa_score']['mean']:,.0f}"
                if d in payload["results"]
                else "–"
                for d in DOMAINS
            ]
        )
    lines += _table(["Experiment", *DOMAINS], rows)

    lines += ["**Selection detail: CV RMSE, chosen parameters, 95% bootstrap CI on test RMSE**", ""]
    rows = []
    for name, payload in results.items():
        for domain in DOMAINS:
            outcome = payload["results"].get(domain)
            if outcome is None:
                continue
            low, high = outcome["test"]["rmse_ci95_seed0"]
            params = ", ".join(f"{k}={v}" for k, v in outcome["selection"]["best_params"].items())
            grid = payload["config"]["model"].get("grid", {})
            if any(
                len(grid[k]) > 1 and v in (min(grid[k]), max(grid[k]))
                for k, v in outcome["selection"]["best_params"].items()
            ):
                params += " (grid edge)"
            rows.append(
                [
                    f"`{name}`",
                    domain,
                    str(outcome["feature_count"]),
                    f"{outcome['cv_rmse']['mean']:.2f} ± {outcome['cv_rmse']['std']:.2f}",
                    params or "–",
                    f"[{low:.1f}, {high:.1f}]",
                ]
            )
    lines += _table(
        ["Experiment", "Data", "Features", "CV RMSE", "Selected", "Test RMSE 95% CI"],
        rows,
        ":---|:---|---:|---:|:---|---:",
    )
    for name, payload in results.items():
        lines.append(f"- `{name}`: {payload['description']}")
    lines.append("")
    return lines


def _calibration_section(reports: Path) -> list[str]:
    calibration = _load(reports / "drift_calibration.json")
    if calibration is None:
        return ["_No calibration results found. Run `make calibrate`._", ""]
    lines = [
        f"- Legacy rule: {calibration['legacy_rule']}.",
        f"- Current rule: {calibration['current_rule']}.",
        f"- {calibration['trials']} Monte Carlo batches per cell.",
        "",
        "**False-alarm rate: synthetic FD001 batch vs FD001 reference (no drift exists)**",
        "",
    ]
    rows = [
        [size, _pct(v["legacy"]), _pct(v["current"]), f"{v['noise_floor']:.3f}"]
        for size, v in calibration["synthetic_false_alarms"].items()
    ]
    lines += _table(["Batch rows", "Legacy", "Current", "Noise floor"], rows)
    real = calibration.get("real")
    if real:
        lines += [
            (
                f"**False-alarm rate on real data** ({real['reference']} as reference, "
                f"{real['holdout']} as batches)"
            ),
            "",
        ]
        rows = []
        for size in real["false_alarms"]:
            fleet = real["false_alarms"][size]
            single = real["false_alarms_single_engine_windows"][size]
            rows.append(
                [size, _pct(fleet["legacy"]), _pct(fleet["current"]), _pct(single["current"])]
            )
        lines += _table(
            ["Batch rows", "Legacy (fleet)", "Current (fleet)", "Current (single-engine window)"],
            rows,
        )
        lines += [
            "Fleet batches draw rows across many engines, like a scoring window over a live",
            "fleet. Consecutive file rows cover one or two engines at a single life stage; their",
            "degradation-correlated sensors differ from the fleet-wide reference, so *any*",
            "distribution test alarms on them. Monitor fleet cross-sections, not single engines.",
            "",
            "**Detection rate of real regime shifts (fleet batches vs FD001 reference)**",
            "",
        ]
        rows = [
            [domain, *(_pct(v["current"]) for v in sizes.values())]
            for domain, sizes in real["detection"].items()
        ]
        sizes = list(next(iter(real["detection"].values())))
        lines += _table(["Shift", *(f"{s} rows" for s in sizes)], rows)
        lines += ["**Power: calibration bias on `sensor_2` in units of reference σ**", ""]
        power = real["power_sensor_2_bias_sigma"]
        levels = list(next(iter(power.values())))
        rows = [
            [f"{size} rows", *(_pct(curve[k]["current"]) for k in levels)]
            for size, curve in power.items()
        ]
        lines += _table(["Batch", *(f"{k}σ" for k in levels)], rows)
        lines += [
            "The 0.20 PSI line is a practical-significance threshold: shifts of about 0.5σ or",
            "more on a single sensor are caught; smaller ones are intentionally tolerated.",
            "",
        ]
    canary = calibration.get("canary")
    if canary:
        lines += [
            (
                "**Canary operating characteristic** (real FD001 shadow errors; tolerance "
                f"{_pct(canary['tolerance'])}): probability of promotion vs the canary's true "
                "MAE change"
            ),
            "",
        ]
        for minimum, rows_data in canary["curves"].items():
            rows = [
                [
                    f"{100 * r['true_relative_mae_change']:+.1f}%",
                    _pct(r["legacy_promote_rate"]),
                    _pct(r["promote_rate"]),
                    f"{r['mean_observations_used']:.0f}",
                ]
                for r in rows_data
            ]
            lines += [f"Minimum {minimum} shadow observations:", ""]
            lines += _table(
                ["True MAE change", "Legacy promote", "Current promote", "Obs. used"], rows
            )
        lines += [
            "The legacy point-estimate rule promotes canaries that are truly worse than the",
            "tolerance about half the time near the boundary; the paired non-inferiority test",
            "only promotes when the confidence bound clears the margin and otherwise collects",
            "more evidence before keeping production.",
            "",
        ]
    return lines


def _synthetic_section(reports: Path) -> list[str]:
    lines: list[str] = []
    matrix = _load(reports / "matrix.json")
    if matrix:
        lines += [
            "**Drift matrix** (synthetic batch vs synthetic FD001 reference; `FD001` is the",
            "negative control)",
            "",
        ]
        rows = [
            [
                domain,
                "yes" if r["expected_detected"] else "no",
                "yes" if r["detected"] else "no",
                f"{r['psi']:.3f}",
                f"{r['max_standardized_shift']:.2f}",
                ", ".join(r["event_kinds"][1:]) or "–",
            ]
            for domain, r in matrix.items()
        ]
        lines += _table(
            ["Scenario", "Expect", "Detected", "PSI", "Max shift (σ)", "Events"],
            rows,
            ":---|:---|:---|---:|---:|:---",
        )
    for scenario in ("promote", "rollback"):
        demo = _load(reports / f"demo_{scenario}.json")
        if demo is None:
            continue
        events = [
            event
            for batch in ("first_batch", "second_batch")
            for event in demo[batch]["events"]
            if event["kind"] in {"promoted", "rolled_back"}
        ]
        detail = (
            f"{events[-1]['kind'].replace('_', ' ')}: {events[-1]['detail']}"
            if events
            else "no release decision"
        )
        lines.append(
            f"- `make {'demo' if scenario == 'promote' else 'rollback'}` → aliases "
            f"`{demo['status']['aliases']}`; {detail}"
        )
    lines.append("")
    return lines


def _benchmark_section(reports: Path) -> list[str]:
    benchmark = _load(reports / "benchmark.json")
    if benchmark is None:
        return ["_No benchmark found. Run `make benchmark`._", ""]
    rows = [
        ["Real rows loaded (FD001 + FD002 train)", f"{benchmark['records_evaluated']:,}"],
        ["Load time", f"{benchmark['load_time_seconds']:.3f} s"],
        ["Bootstrap on FD001 (fit + reference)", f"{benchmark['bootstrap_time_seconds']:.3f} s"],
        ["500-row FD002 batch: score + retrain + canary", f"{benchmark['batch_inference_and_drift_time_seconds']:.3f} s"],
        ["Next 500-row batch", f"{benchmark['canary_evaluation_time_seconds']:.3f} s"],
        ["Drift PSI on first FD002 batch", f"{benchmark['drift_psi']:.3f}"],
        ["Events", ", ".join(benchmark["events"])],
    ]
    return _table(["Measure", "Value"], rows, ":---|:---")


def render(reports: Path, reference_dir: Path) -> str:
    manifest = _load(reference_dir / "MANIFEST.json")
    lines = [
        "# Evaluation",
        "",
        "<!-- Generated by `make report` from reports/*.json. Do not edit by hand. -->",
        "",
        "All numbers below are regenerated by `make report` (which runs `make data eval calibrate`",
        "first). Synthetic results exercise the control loop; real NASA C-MAPSS results measure the",
        "data science. Runtime figures depend on the machine that generated them.",
        "",
        "## 1. RUL model quality on real NASA C-MAPSS",
        "",
        *_experiments_section(reports),
        "## 2. Drift detector and canary calibration",
        "",
        *_calibration_section(reports),
        "## 3. Closed-loop checks on synthetic telemetry",
        "",
        *_synthetic_section(reports),
        "## 4. Runtime on real data",
        "",
        *_benchmark_section(reports),
        "## 5. Data provenance",
        "",
    ]
    if manifest:
        lines += [f"Source mirror: {manifest['source']} (verified by `make data`).", ""]
        rows = [
            [name, f"{entry['rows']:,}", f"`{entry['sha256'][:16]}…`"]
            for name, entry in manifest["files"].items()
        ]
        lines += _table(["File", "Rows", "SHA-256"], rows, ":---|---:|:---")
    lines += [
        "## 6. Known limitations",
        "",
        "- The live loop still serves the 9-feature linear contract (`serving_ridge`); the offline",
        "  feature set and kernel model are evaluated here but not yet wired into serving.",
        "- Rows within an engine are autocorrelated, so the chi-square noise floor (which assumes",
        "  independent rows) slightly under-states real false alarms for small batches (compare",
        "  the fleet column in §2 with the 1% design rate); prefer batches of ≥200 rows.",
        "- Canary calibration simulates candidate quality on real residuals rather than training",
        "  many real candidates.",
        "",
    ]
    return "\n".join(lines)


def write_report(reports: Path, reference_dir: Path, destination: Path) -> Path:
    destination.write_text(render(reports, reference_dir), encoding="utf-8")
    return destination
