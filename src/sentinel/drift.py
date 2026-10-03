from __future__ import annotations

from statistics import NormalDist

import numpy as np

from sentinel.features import FEATURE_NAMES
from sentinel.types import DriftReport

# Jeffreys-style pseudo-count added to every bin. A hard floor (e.g. clip at 1e-6) lets one empty
# bin in a small batch contribute ~1.5 PSI on its own and makes every large shift saturate at the
# same ceiling; additive smoothing keeps PSI finite and comparable across batch sizes.
PSEUDO_COUNT = 0.5


def chi2_quantile(probability: float, degrees: int) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (accurate to ~1% for df >= 3)."""

    z = NormalDist().inv_cdf(probability)
    k = float(degrees)
    return k * (1 - 2 / (9 * k) + z * (2 / (9 * k)) ** 0.5) ** 3


class PSIDriftDetector:
    """Owns the reference distribution and scores a complete batch in one call.

    The aggregate score is the maximum per-feature PSI. Under no drift, PSI between a batch of n
    rows and a reference of m rows is approximately chi2(bins - 1) * (1/n + 1/m), so a fixed
    threshold that is safe for large batches false-alarms on small ones. Detection therefore
    requires PSI to clear both the configured `threshold` (practical significance) and a
    sampling-noise floor at family-wise false-alarm rate `false_alarm_rate` across all features
    (statistical significance). See `sentinel.evaluation.drift_calibration` for the measured rates.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        bins: int = 8,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        warning_threshold: float | None = None,
        false_alarm_rate: float | None = 0.01,
    ) -> None:
        self.threshold = threshold
        self.warning_threshold = threshold / 2 if warning_threshold is None else warning_threshold
        self.bins = bins
        self.feature_names = feature_names
        self.false_alarm_rate = false_alarm_rate
        self._reference_rows = 0
        self._edges: list[np.ndarray] | None = None
        self._expected: list[np.ndarray] | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._edges is not None

    def fit_reference(self, x: np.ndarray) -> None:
        if x.ndim != 2 or len(x) < self.bins:
            raise ValueError("reference requires more rows than PSI bins")
        if x.shape[1] != len(self.feature_names):
            raise ValueError("reference feature shape differs from monitored feature names")
        edges: list[np.ndarray] = []
        expected: list[np.ndarray] = []
        quantiles = np.linspace(0, 1, self.bins + 1)
        for column in x.T:
            feature_edges = np.unique(np.quantile(column, quantiles))
            if len(feature_edges) < 3:
                center = float(column.mean())
                feature_edges = np.array([center - 1e-6, center, center + 1e-6])
            feature_edges[0] = -np.inf
            feature_edges[-1] = np.inf
            counts, _ = np.histogram(column, bins=feature_edges)
            edges.append(feature_edges)
            expected.append(self._proportions(counts))
        self._edges = edges
        self._expected = expected
        self._reference_rows = len(x)
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        self._scale = scale

    def score(self, x: np.ndarray) -> DriftReport:
        if self._edges is None or self._expected is None:
            raise RuntimeError("fit_reference must be called before score")
        if x.ndim != 2 or x.shape[1] != len(self._edges):
            raise ValueError("batch feature shape differs from reference")
        scores: dict[str, float] = {}
        for name, column, edges, expected in zip(
            self.feature_names, x.T, self._edges, self._expected, strict=True
        ):
            actual_counts, _ = np.histogram(column, bins=edges)
            actual = self._proportions(actual_counts)
            scores[name] = float(np.sum((actual - expected) * np.log(actual / expected)))
        # Standardized mean difference: an unbounded effect size that keeps ranking severity
        # after PSI approaches its binned ceiling.
        shift = np.abs(x.mean(axis=0) - self._mean) / self._scale
        shifts = {name: float(value) for name, value in zip(self.feature_names, shift, strict=True)}
        aggregate = max(scores.values())
        floor = self.noise_floor(len(x))
        alert_line = max(self.threshold, floor)
        return DriftReport(
            detected=aggregate >= alert_line,
            aggregate_psi=aggregate,
            feature_psi=scores,
            threshold=self.threshold,
            feature_shift=shifts,
            warning=max(self.warning_threshold, floor) <= aggregate < alert_line,
            noise_floor=floor,
        )

    def noise_floor(self, batch_rows: int) -> float:
        """PSI that sampling noise alone exceeds with probability `false_alarm_rate`."""

        if self.false_alarm_rate is None or batch_rows < 1 or self._reference_rows < 1:
            return 0.0
        per_feature = self.false_alarm_rate / len(self.feature_names)
        quantile = chi2_quantile(1 - per_feature, self.bins - 1)
        return quantile * (1 / batch_rows + 1 / self._reference_rows)

    @staticmethod
    def _proportions(counts: np.ndarray) -> np.ndarray:
        smoothed = counts.astype(float) + PSEUDO_COUNT
        return smoothed / smoothed.sum()
