"""Monte Carlo calibration of the drift detector and the canary decision rule.

Every study compares the rule as served today (`PSIDriftDetector` with its sampling-noise floor,
`non_inferiority_decision`) against the legacy rule it replaced (fixed PSI >= 0.20 over all 9
features with a 1e-6 bin floor; point-estimate MAE comparison), on synthetic and real data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sentinel.datasets import load_cmapps_arrays
from sentinel.drift import PSIDriftDetector
from sentinel.evaluation.experiment import ServingFeatureBuilder
from sentinel.features import FEATURE_NAMES, MONITORED_FEATURES, MONITORED_INDEX
from sentinel.model import RidgeModel
from sentinel.router import non_inferiority_decision
from sentinel.synthetic import generate_telemetry

LEGACY_THRESHOLD = 0.2
BATCH_SIZES = (50, 100, 200, 400)


class LegacyPSI:
    """The pre-calibration detector: 9 features incl. cycle, 1e-6 floor, fixed 0.20 threshold."""

    def __init__(self, reference: np.ndarray, bins: int = 8) -> None:
        self.edges, self.expected = [], []
        for column in reference.T:
            edges = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                center = float(column.mean())
                edges = np.array([center - 1e-6, center, center + 1e-6])
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges.append(edges)
            self.expected.append(self._proportions(np.histogram(column, bins=edges)[0]))

    @staticmethod
    def _proportions(counts: np.ndarray) -> np.ndarray:
        return np.clip(counts / max(1, counts.sum()), 1e-6, None)

    def score(self, batch: np.ndarray) -> float:
        scores = []
        for column, edges, expected in zip(batch.T, self.edges, self.expected, strict=True):
            actual = self._proportions(np.histogram(column, bins=edges)[0])
            scores.append(float(np.sum((actual - expected) * np.log(actual / expected))))
        return max(scores)


def _detectors(reference: np.ndarray) -> tuple[PSIDriftDetector, LegacyPSI]:
    current = PSIDriftDetector(threshold=0.2, feature_names=MONITORED_FEATURES)
    current.fit_reference(reference[:, MONITORED_INDEX])
    return current, LegacyPSI(reference)


def _rates(
    reference: np.ndarray, batches: list[np.ndarray]
) -> dict[str, float]:
    current, legacy = _detectors(reference)
    hits_current = [current.score(b[:, MONITORED_INDEX]).detected for b in batches]
    hits_legacy = [legacy.score(b) >= LEGACY_THRESHOLD for b in batches]
    return {
        "current": float(np.mean(hits_current)),
        "legacy": float(np.mean(hits_legacy)),
        "noise_floor": current.noise_floor(len(batches[0])),
        "trials": len(batches),
    }


def _windows(features: np.ndarray, size: int, trials: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Consecutive file rows: one or two engines at a single life stage (the pitfall case)."""

    starts = rng.integers(0, len(features) - size, size=trials)
    return [features[start : start + size] for start in starts]


def _fleet(features: np.ndarray, size: int, trials: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Fleet cross-section: rows drawn across many engines and life stages (production-like)."""

    return [features[rng.choice(len(features), size=size, replace=False)] for _ in range(trials)]


def _synthetic_features(domain: str, rows: int, seed: int, offset: int) -> np.ndarray:
    from sentinel.features import RollingFeaturePipeline

    telemetry = generate_telemetry(domain, math.ceil(rows / 35), 35, seed=seed, engine_offset=offset)
    features = RollingFeaturePipeline().transform(telemetry[:rows])
    return np.asarray([row.values for row in features])


def synthetic_false_alarms(trials: int = 200) -> dict[str, object]:
    """FD001 vs FD001 (same generator, new seeds): every alarm is a false alarm."""

    reference = _synthetic_features("FD001", 360, seed=10, offset=0)
    return {
        str(size): _rates(
            reference,
            [_synthetic_features("FD001", size, seed=1000 + t, offset=500) for t in range(trials)],
        )
        for size in BATCH_SIZES
    }


def _real_features(reference_dir: Path, domain: str) -> tuple[np.ndarray, np.ndarray]:
    data = load_cmapps_arrays(reference_dir / f"train_{domain}.txt", domain)
    return ServingFeatureBuilder().transform(data), data.engine


def real_studies(reference_dir: Path, trials: int = 200, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    fd001, engines = _real_features(reference_dir, "FD001")
    in_reference = engines <= 70
    reference, holdout = fd001[in_reference], fd001[~in_reference]

    false_alarms = {
        str(size): _rates(reference, _fleet(holdout, size, trials, rng)) for size in BATCH_SIZES
    }
    single_engine_windows = {
        str(size): _rates(reference, _windows(holdout, size, trials, rng)) for size in BATCH_SIZES
    }

    detection = {}
    for domain in ("FD002", "FD003", "FD004"):
        shifted, _ = _real_features(reference_dir, domain)
        detection[domain] = {
            str(size): _rates(reference, _fleet(shifted, size, trials, rng))
            for size in BATCH_SIZES
        }

    # Power: inject a calibration bias of k reference standard deviations into sensor_2 only.
    column = FEATURE_NAMES.index("sensor_2")
    sigma = float(reference[:, column].std())
    power = {}
    for size in (100, 400):
        curve = {}
        for k in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
            batches = []
            for window in _fleet(holdout, size, trials, rng):
                biased = window.copy()
                biased[:, column] += k * sigma
                batches.append(biased)
            curve[str(k)] = _rates(reference, batches)
        power[str(size)] = curve
    return {
        "reference": "FD001 engines 1-70",
        "holdout": "FD001 engines 71-100",
        "batch_sampling": "fleet cross-section (rows drawn across holdout engines)",
        "false_alarms": false_alarms,
        "false_alarms_single_engine_windows": single_engine_windows,
        "detection": detection,
        "power_sensor_2_bias_sigma": power,
    }


def canary_operating_characteristic(
    reference_dir: Path, trials: int = 200, tolerance: float = 0.03, seed: int = 0
) -> dict[str, object]:
    """Promotion probability as a function of the canary's true relative MAE change.

    Production is the serving ridge trained on FD001 engines 1-70; its holdout errors e_i are
    real. A canary with quality lambda has errors (1 - lambda) * e_i + eta_i, with eta_i an
    independent model disagreement term, so its true MAE change is measured on the full holdout.
    """

    rng = np.random.default_rng(seed)
    features, engines = _real_features(reference_dir, "FD001")
    data = load_cmapps_arrays(reference_dir / "train_FD001.txt", "FD001")
    target = np.minimum(data.rul, 125.0)
    train = engines <= 70
    model = RidgeModel.fit(features[train], target[train])
    residual = model.predict(features[~train]) - target[~train]
    production_errors = np.abs(residual)
    disagreement = rng.normal(0.0, 0.25 * residual.std(), size=residual.shape)
    curves = {}
    for min_observations in (50, 150):
        rows = []
        for quality in (0.3, 0.15, 0.08, 0.04, 0.0, -0.04, -0.08, -0.15, -0.3):
            canary_errors = np.abs((1 - quality) * residual + disagreement)
            true_change = float(canary_errors.mean() / production_errors.mean() - 1)
            outcomes = {"promoted": 0, "rolled_back": 0}
            legacy_promotions = 0
            used = []
            for _ in range(trials):
                order = rng.permutation(len(residual))[: 4 * min_observations]
                p, c = production_errors[order], canary_errors[order]
                outcome = None
                for n in range(min_observations, len(order) + 1):
                    outcome, _ = non_inferiority_decision(
                        p[:n], c[:n], tolerance, min_observations, 4 * min_observations
                    )
                    if outcome is not None:
                        used.append(n)
                        break
                outcomes[outcome] += 1
                head = slice(0, min_observations)
                legacy_promotions += c[head].mean() <= p[head].mean() * (1 + tolerance)
            rows.append(
                {
                    "true_relative_mae_change": true_change,
                    "promote_rate": outcomes["promoted"] / trials,
                    "legacy_promote_rate": legacy_promotions / trials,
                    "mean_observations_used": float(np.mean(used)),
                }
            )
        curves[str(min_observations)] = rows
    return {"tolerance": tolerance, "trials": trials, "curves": curves}


def run_calibration(reference_dir: Path, trials: int = 200) -> dict[str, object]:
    result: dict[str, object] = {
        "trials": trials,
        "legacy_rule": "max PSI over 9 features incl. cycle_scaled, 1e-6 bin floor, >= 0.20",
        "current_rule": (
            "max PSI over 8 monitored features, Jeffreys smoothing, >= max(0.20, chi-square "
            "noise floor at 1% family-wise false-alarm rate)"
        ),
        "synthetic_false_alarms": synthetic_false_alarms(trials),
    }
    if (reference_dir / "train_FD004.txt").exists():
        result["real"] = real_studies(reference_dir, trials)
        result["canary"] = canary_operating_characteristic(reference_dir, trials)
    return result
