from __future__ import annotations

import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def nasa_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """PHM08 asymmetric score: late predictions (overestimating RUL) are penalized harder."""

    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return float(np.sum(np.where(error < 0, np.exp(-error / 13.0), np.exp(error / 10.0)) - 1.0))


def regression_report(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
        "nasa_score": nasa_score(prediction, target),
        "count": float(len(target)),
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic of one sample."""

    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = np.apply_along_axis(statistic, 1, values[draws])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(low), float(high)
