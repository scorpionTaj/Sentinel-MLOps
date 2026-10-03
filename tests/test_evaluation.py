from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sentinel.datasets import CMAPSSArrays, load_cmapps_arrays
from sentinel.drift import PSIDriftDetector, chi2_quantile
from sentinel.evaluation.cmapss import FeatureConfig, OfflineFeatureBuilder, rolling_statistics
from sentinel.evaluation.metrics import bootstrap_ci, mae, nasa_score, rmse
from sentinel.evaluation.models import MODELS, build_model
from sentinel.evaluation.splits import group_kfold, group_split
from sentinel.router import non_inferiority_decision


def _arrays(engines: int = 6, cycles: int = 40, regimes: int = 1, seed: int = 0) -> CMAPSSArrays:
    rng = np.random.default_rng(seed)
    engine = np.repeat(np.arange(1, engines + 1), cycles)
    cycle = np.tile(np.arange(1, cycles + 1), engines)
    regime = rng.integers(0, regimes, size=len(engine))
    settings = np.column_stack([regime * 10.0, regime * 0.2, np.full(len(engine), 100.0)])
    health = 1 - cycle / cycles
    sensors = rng.normal(0, 0.01, size=(len(engine), 21)) + 500.0
    sensors[:, 1] += 5 * health + regime * 40
    sensors[:, 2] -= 3 * health
    sensors[:, 0] = 518.67  # constant sensor, as in FD001
    return CMAPSSArrays("FD001", engine, cycle, settings, sensors, (cycles - cycle).astype(float))


class MetricTests(unittest.TestCase):
    def test_point_metrics(self) -> None:
        prediction, target = np.array([10.0, 20.0, 30.0]), np.array([12.0, 20.0, 27.0])
        self.assertAlmostEqual(mae(prediction, target), 5 / 3)
        self.assertAlmostEqual(rmse(prediction, target), np.sqrt(13 / 3))

    def test_nasa_score_penalizes_late_predictions_more(self) -> None:
        early = nasa_score(np.array([90.0]), np.array([100.0]))
        late = nasa_score(np.array([110.0]), np.array([100.0]))
        self.assertAlmostEqual(early, np.exp(10 / 13) - 1)
        self.assertAlmostEqual(late, np.exp(1.0) - 1)
        self.assertGreater(late, early)

    def test_bootstrap_ci_brackets_the_mean(self) -> None:
        values = np.random.default_rng(1).normal(5, 1, size=400)
        low, high = bootstrap_ci(values, resamples=500)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertLess(high - low, 0.5)


class SplitTests(unittest.TestCase):
    def test_group_split_never_shares_an_engine(self) -> None:
        groups = np.repeat(np.arange(10), 7)
        train, validation = group_split(groups, 0.3)
        self.assertFalse(set(groups[train]) & set(groups[validation]))
        self.assertEqual(len(set(groups[validation])), 3)

    def test_single_group_falls_back_to_chronological_split(self) -> None:
        train, validation = group_split(["a"] * 10, 0.3)
        self.assertEqual(list(train), list(range(7)))
        self.assertEqual(list(validation), [7, 8, 9])

    def test_group_kfold_partitions_rows_without_leakage(self) -> None:
        groups = np.repeat(np.arange(12), 5)
        seen = []
        for train, validation in group_kfold(groups, folds=4, seed=3):
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            seen.extend(validation)
        self.assertEqual(sorted(seen), list(range(len(groups))))


class OfflineFeatureTests(unittest.TestCase):
    def test_rolling_statistics_are_causal_and_reset_per_engine(self) -> None:
        values = np.array([[1.0], [2.0], [3.0], [10.0], [20.0]])
        engine = np.array([1, 1, 1, 2, 2])
        mean, slope, std = rolling_statistics(values, engine, window=2)
        np.testing.assert_allclose(mean[:, 0], [1.0, 1.5, 2.5, 10.0, 15.0])
        np.testing.assert_allclose(slope[:, 0], [0.0, 1.0, 1.0, 0.0, 10.0])
        np.testing.assert_allclose(std[:, 0], [0.0, 0.5, 0.5, 0.0, 5.0])

    def test_builder_drops_constant_sensors_and_normalizes_per_regime(self) -> None:
        data = _arrays(regimes=3)
        builder = OfflineFeatureBuilder(FeatureConfig(regimes=3, rolling=())).fit(data)
        self.assertNotIn("s1", builder.names)
        features = builder.transform(data)
        column = builder.names.index("s2")
        # The +40 per-regime offset is removed; only the degradation trend remains.
        self.assertLess(abs(np.corrcoef(features[:, column], data.settings[:, 0])[0, 1]), 0.2)

    def test_builder_uses_training_statistics_only(self) -> None:
        train, shifted = _arrays(seed=0), _arrays(seed=1)
        shifted.sensors[:, 2] += 100.0
        builder = OfflineFeatureBuilder(FeatureConfig(rolling=())).fit(train)
        column = builder.names.index("s3")
        self.assertGreater(builder.transform(shifted)[:, column].mean(), 50)

    def test_every_model_learns_a_linear_trend(self) -> None:
        x = np.linspace(0, 1, 200)[:, None]
        y = 100 * x[:, 0]
        for kind in MODELS:
            params = {"alpha": 0.01} if kind != "mean" else {}
            model = build_model(kind, params, seed=0).fit(x, y)
            error = rmse(model.predict(x), y)
            with self.subTest(model=kind):
                self.assertLess(error, 30 if kind == "mean" else 3)


class LoaderTests(unittest.TestCase):
    def test_array_loader_matches_terminal_rul_offsets(self) -> None:
        rows = []
        for engine, cycles in ((1, 3), (2, 2)):
            for cycle in range(1, cycles + 1):
                rows.append(" ".join([str(engine), str(cycle)] + ["1"] * 24))
        with tempfile.TemporaryDirectory() as temporary:
            test_path, rul_path = Path(temporary) / "t.txt", Path(temporary) / "r.txt"
            test_path.write_text("\n".join(rows), encoding="utf-8")
            rul_path.write_text("10\n5\n", encoding="utf-8")
            data = load_cmapps_arrays(test_path, "FD001", rul_path)
        np.testing.assert_allclose(data.rul, [12, 11, 10, 6, 5])
        self.assertEqual(list(data.last_cycle_mask()), [False, False, True, False, True])


class DetectorCalibrationTests(unittest.TestCase):
    def test_chi2_quantile_approximation(self) -> None:
        self.assertAlmostEqual(chi2_quantile(0.95, 7), 14.067, delta=0.1)
        self.assertAlmostEqual(chi2_quantile(0.99875, 7), 24.3, delta=0.4)

    def test_noise_floor_shrinks_with_batch_size(self) -> None:
        detector = PSIDriftDetector()
        detector.fit_reference(np.random.default_rng(0).normal(size=(1000, 9)))
        self.assertGreater(detector.noise_floor(50), detector.noise_floor(500))
        self.assertGreater(detector.noise_floor(50), detector.threshold)

    def test_small_stable_batches_rarely_alarm(self) -> None:
        rng = np.random.default_rng(5)
        detector = PSIDriftDetector()
        detector.fit_reference(rng.normal(size=(400, 9)))
        alarms = sum(detector.score(rng.normal(size=(50, 9))).detected for _ in range(200))
        self.assertLessEqual(alarms, 6)

    def test_non_inferiority_requires_confidence(self) -> None:
        production = np.full(100, 10.0) + np.random.default_rng(0).normal(0, 2, 100)
        better, _ = non_inferiority_decision(production, production * 0.8, 0.03, 50, 200)
        worse, _ = non_inferiority_decision(production, production * 1.3, 0.03, 50, 200)
        early, _ = non_inferiority_decision(production[:10], production[:10], 0.03, 50, 200)
        self.assertEqual((better, worse, early), ("promoted", "rolled_back", None))


if __name__ == "__main__":
    unittest.main()
