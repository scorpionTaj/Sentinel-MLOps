"""Offline C-MAPSS feature engineering, fitted on training engines only.

This is the research feature set used to evaluate modelling choices on real data. It is
deliberately separate from `sentinel.features.RollingFeaturePipeline`, which defines the 9-value
serving contract of the live loop and API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from sentinel.datasets import CMAPSSArrays


def kmeans(x: np.ndarray, k: int, seed: int = 0, iterations: int = 50) -> np.ndarray:
    """Deterministic k-means++ returning centroids; used to recover flight-condition regimes."""

    rng = np.random.default_rng(seed)
    centroids = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        distance = np.min([np.sum((x - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(x[rng.choice(len(x), p=distance / distance.sum())])
    centers = np.asarray(centroids, dtype=float)
    for _ in range(iterations):
        labels = assign(x, centers)
        updated = np.array(
            [x[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


def rolling_statistics(
    values: np.ndarray, engine: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-engine rolling mean, least-squares slope and standard deviation.

    Each row only sees itself and up to `window - 1` earlier cycles of the same engine.
    """

    mean = np.empty_like(values)
    slope = np.zeros_like(values)
    std = np.zeros_like(values)
    boundaries = np.flatnonzero(np.r_[True, engine[1:] != engine[:-1], True])
    for start, stop in pairwise(boundaries):
        y = values[start:stop]
        t = np.arange(stop - start, dtype=float)[:, None]

        def windowed(series: np.ndarray) -> np.ndarray:
            total = np.cumsum(series, axis=0)
            shifted = np.zeros_like(total)
            shifted[window:] = total[:-window]
            return total - shifted

        count = np.minimum(np.arange(1, len(y) + 1), window)[:, None].astype(float)
        sum_y, sum_ty = windowed(y), windowed(t * y)
        sum_t, sum_tt = windowed(t), windowed(t * t)
        sum_yy = windowed(y * y)
        mean[start:stop] = sum_y / count
        denominator = count * sum_tt - sum_t**2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[start:stop] = np.where(
                denominator > 0, (count * sum_ty - sum_t * sum_y) / denominator, 0.0
            )
        variance = np.maximum(sum_yy / count - (sum_y / count) ** 2, 0.0)
        std[start:stop] = np.sqrt(variance)
    return mean, slope, std


@dataclass
class FeatureConfig:
    regimes: int = 1
    window: int = 30
    rolling: tuple[str, ...] = ("mean", "slope", "std")
    include_raw: bool = True
    include_cycle: bool = True
    min_distinct_values: int = 10


@dataclass
class OfflineFeatureBuilder:
    """Regime-aware normalization plus causal rolling features for C-MAPSS arrays."""

    config: FeatureConfig = field(default_factory=FeatureConfig)
    centers: np.ndarray | None = None
    sensor_mean: np.ndarray | None = None
    sensor_scale: np.ndarray | None = None
    kept_sensors: np.ndarray | None = None
    names: list[str] = field(default_factory=list)

    def fit(self, data: CMAPSSArrays) -> OfflineFeatureBuilder:
        settings = self._scaled_settings(data)
        k = self.config.regimes
        self.centers = kmeans(settings, k) if k > 1 else settings.mean(axis=0, keepdims=True)
        labels = assign(settings, self.centers)
        means = np.zeros((k, data.sensors.shape[1]))
        scales = np.ones((k, data.sensors.shape[1]))
        distinct = np.zeros(data.sensors.shape[1], dtype=int)
        for regime in range(k):
            block = data.sensors[labels == regime]
            means[regime] = block.mean(axis=0)
            std = block.std(axis=0)
            distinct = np.maximum(distinct, [len(np.unique(column)) for column in block.T])
            scales[regime] = np.where(std > 1e-9, std, 1.0)
        self.sensor_mean, self.sensor_scale = means, scales
        # Channels that only take a handful of quantized values inside every operating regime
        # (constant or flickering between two readings) carry no degradation signal.
        self.kept_sensors = np.flatnonzero(distinct >= self.config.min_distinct_values)
        sensor_names = [f"s{index + 1}" for index in self.kept_sensors]
        self.names = (["cycle"] if self.config.include_cycle else []) + (
            sensor_names if self.config.include_raw else []
        )
        for statistic in self.config.rolling:
            self.names += [f"{name}_{statistic}{self.config.window}" for name in sensor_names]
        return self

    def transform(self, data: CMAPSSArrays) -> np.ndarray:
        if self.centers is None or self.kept_sensors is None:
            raise RuntimeError("fit must be called before transform")
        labels = assign(self._scaled_settings(data), self.centers)
        normalized = (data.sensors - self.sensor_mean[labels]) / self.sensor_scale[labels]
        normalized = normalized[:, self.kept_sensors]
        columns = []
        if self.config.include_cycle:
            columns.append(data.cycle[:, None].astype(float))
        if self.config.include_raw:
            columns.append(normalized)
        if self.config.rolling:
            mean, slope, std = rolling_statistics(normalized, data.engine, self.config.window)
            lookup = {"mean": mean, "slope": slope, "std": std}
            columns.extend(lookup[name] for name in self.config.rolling)
        return np.column_stack(columns)

    @staticmethod
    def _scaled_settings(data: CMAPSSArrays) -> np.ndarray:
        # Altitude (~0-42 kft), Mach (~0-0.84) and throttle (~60-100) live on different scales.
        return data.settings / np.array([42.0, 0.84, 100.0])
