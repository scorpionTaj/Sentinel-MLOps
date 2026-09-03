from __future__ import annotations

import numpy as np

from sentinel.features import FEATURE_NAMES
from sentinel.types import DriftReport


class PSIDriftDetector:
    """Owns the reference distribution and scores a complete batch in one call."""

    def __init__(self, threshold: float = 0.2, bins: int = 8) -> None:
        self.threshold = threshold
        self.bins = bins
        self._edges: list[np.ndarray] | None = None
        self._expected: list[np.ndarray] | None = None

    def fit_reference(self, x: np.ndarray) -> None:
        if x.ndim != 2 or len(x) < self.bins:
            raise ValueError("reference requires more rows than PSI bins")
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

    def score(self, x: np.ndarray) -> DriftReport:
        if self._edges is None or self._expected is None:
            raise RuntimeError("fit_reference must be called before score")
        if x.ndim != 2 or x.shape[1] != len(self._edges):
            raise ValueError("batch feature shape differs from reference")
        scores: dict[str, float] = {}
        for name, column, edges, expected in zip(
            FEATURE_NAMES, x.T, self._edges, self._expected, strict=True
        ):
            actual_counts, _ = np.histogram(column, bins=edges)
            actual = self._proportions(actual_counts)
            value = float(np.sum((actual - expected) * np.log(actual / expected)))
            scores[name] = value
        aggregate = max(scores.values())
        return DriftReport(aggregate >= self.threshold, aggregate, scores, self.threshold)

    @staticmethod
    def _proportions(counts: np.ndarray) -> np.ndarray:
        proportions = counts.astype(float) / max(1, int(counts.sum()))
        return np.clip(proportions, 1e-6, None)

