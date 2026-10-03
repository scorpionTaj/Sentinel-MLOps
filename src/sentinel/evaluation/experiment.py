"""Config-driven, leakage-free RUL experiments on real NASA C-MAPSS data.

Protocol (per dataset):
1. Hyperparameters are chosen by engine-grouped K-fold CV on the *training* file only; feature
   normalization and regime clustering are re-fitted inside every fold.
2. The selected configuration is refitted on all training engines for each seed.
3. It is scored once on the official test file at each engine's last observed cycle against
   `RUL_FD00x.txt` (the standard C-MAPSS protocol), with the target capped at `rul_cap`.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.datasets import CMAPSSArrays, load_cmapps_arrays
from sentinel.evaluation.cmapss import FeatureConfig, OfflineFeatureBuilder
from sentinel.evaluation.metrics import bootstrap_ci, mae, nasa_score, rmse
from sentinel.evaluation.models import build_model
from sentinel.evaluation.splits import group_kfold
from sentinel.features import FEATURE_NAMES, RollingFeaturePipeline
from sentinel.types import TelemetryRow

MULTI_CONDITION = {"FD002", "FD004"}


@dataclass
class ServingFeatureBuilder:
    """The live loop's 9-feature contract, built through the real serving pipeline."""

    names: tuple[str, ...] = FEATURE_NAMES

    def fit(self, data: CMAPSSArrays) -> ServingFeatureBuilder:
        return self

    def transform(self, data: CMAPSSArrays) -> np.ndarray:
        selected = data.sensors[:, [1, 2, 3, 6, 10]]
        rows = [
            TelemetryRow(int(e), int(c), float(op), tuple(map(float, s)), float(r), data.domain)
            for e, c, op, s, r in zip(
                data.engine, data.cycle, data.settings[:, 0], selected, data.rul, strict=True
            )
        ]
        return np.asarray([row.values for row in RollingFeaturePipeline().transform(rows)])


def make_builder(spec: dict[str, object], domain: str) -> object:
    if spec.get("set", "offline") == "serving":
        return ServingFeatureBuilder()
    regimes = spec.get("regimes", "auto")
    if regimes == "auto":
        regimes = 6 if domain in MULTI_CONDITION else 1
    return OfflineFeatureBuilder(
        FeatureConfig(
            regimes=int(regimes),
            window=int(spec.get("window", 30)),
            rolling=tuple(spec.get("rolling", ("mean", "slope", "std"))),
            include_raw=bool(spec.get("include_raw", True)),
        )
    )


def _grid(spec: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = sorted(spec)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(spec[k] for k in keys))]


def _cap(y: np.ndarray, cap: float | None) -> np.ndarray:
    return y if cap is None else np.minimum(y, cap)


def _clip(prediction: np.ndarray, cap: float | None) -> np.ndarray:
    return np.clip(prediction, 0.0, cap if cap is not None else None)


def evaluate_dataset(config: dict[str, object], domain: str, reference_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    cap = config.get("rul_cap")
    model_spec = config["model"]
    train = load_cmapps_arrays(reference_dir / f"train_{domain}.txt", domain)
    test = load_cmapps_arrays(
        reference_dir / f"test_{domain}.txt", domain, reference_dir / f"RUL_{domain}.txt"
    )
    target = _cap(train.rul, cap)
    candidates = _grid(model_spec.get("grid", {})) or [{}]
    fold_scores: list[list[float]] = [[] for _ in candidates]
    for fold_train, fold_valid in group_kfold(train.engine, int(config.get("cv_folds", 5)), seed=0):
        builder = make_builder(config["features"], domain).fit(train.subset(fold_train))
        x_train = builder.transform(train.subset(fold_train))
        x_valid = builder.transform(train.subset(fold_valid))
        for index, params in enumerate(candidates):
            model = build_model(model_spec["type"], params, seed=0).fit(x_train, target[fold_train])
            prediction = _clip(model.predict(x_valid), cap)
            fold_scores[index].append(rmse(prediction, target[fold_valid]))
    cv_means = [float(np.mean(scores)) for scores in fold_scores]
    best = int(np.argmin(cv_means))

    builder = make_builder(config["features"], domain).fit(train)
    x_train = builder.transform(train)
    last = test.last_cycle_mask()
    x_test = builder.transform(test)[last]
    true_rul = test.rul[last]
    y_test = _cap(true_rul, cap)
    per_seed = []
    seed_zero_prediction = None
    for seed in config.get("seeds", [0]):
        model = build_model(model_spec["type"], candidates[best], seed=int(seed)).fit(x_train, target)
        prediction = _clip(model.predict(x_test), cap)
        if seed_zero_prediction is None:
            seed_zero_prediction = prediction
        per_seed.append(
            {
                "seed": int(seed),
                "rmse": rmse(prediction, y_test),
                "mae": mae(prediction, y_test),
                "nasa_score": nasa_score(prediction, y_test),
                "rmse_uncapped_target": rmse(prediction, true_rul),
            }
        )
    squared = (seed_zero_prediction - y_test) ** 2
    low, high = bootstrap_ci(squared, resamples=2000, seed=0)

    def summary(key: str) -> dict[str, float]:
        values = np.array([item[key] for item in per_seed])
        return {"mean": float(values.mean()), "std": float(values.std())}

    return {
        "domain": domain,
        "train_rows": len(train),
        "train_engines": len(np.unique(train.engine)),
        "test_engines": int(last.sum()),
        "feature_count": int(x_train.shape[1]),
        "selection": {
            "candidates": [
                {"params": params, "cv_rmse_mean": cv_means[i], "cv_rmse_std": float(np.std(fold_scores[i]))}
                for i, params in enumerate(candidates)
            ],
            "best_params": candidates[best],
        },
        "cv_rmse": {"mean": cv_means[best], "std": float(np.std(fold_scores[best]))},
        "test": {
            "rmse": summary("rmse"),
            "mae": summary("mae"),
            "nasa_score": summary("nasa_score"),
            "rmse_uncapped_target": summary("rmse_uncapped_target"),
            "rmse_ci95_seed0": [float(np.sqrt(low)), float(np.sqrt(high))],
            "per_seed": per_seed,
        },
        "predictions_seed0": {
            "true_rul": [float(v) for v in true_rul],
            "predicted": [float(v) for v in seed_zero_prediction],
        },
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def run_experiment(config_path: Path, reference_dir: Path, out_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = reference_dir / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))["files"] if manifest_path.exists() else {}
    )
    results = {
        "name": config["name"],
        "description": config.get("description", ""),
        "config": config,
        "data_sha256": {
            name: entry["sha256"]
            for name, entry in manifest.items()
            if any(domain in name for domain in config["datasets"])
        },
        "results": {
            domain: evaluate_dataset(config, domain, reference_dir) for domain in config["datasets"]
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{config['name']}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results
