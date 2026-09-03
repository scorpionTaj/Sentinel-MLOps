from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float = 0.5) -> RidgeModel:
        if x.ndim != 2 or len(x) < 2 or len(y) != len(x):
            raise ValueError("training data must be a non-empty feature matrix")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (x - mean) / scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            coefficients=tuple(float(value) for value in weights[1:]),
            intercept=float(weights[0]),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        coefficients = np.asarray(self.coefficients)
        return self.intercept + ((x - mean) / scale) @ coefficients

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RidgeModel:
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),  # type: ignore[arg-type]
            scale=tuple(float(value) for value in payload["scale"]),  # type: ignore[arg-type]
            coefficients=tuple(float(value) for value in payload["coefficients"]),  # type: ignore[arg-type]
            intercept=float(payload["intercept"]),  # type: ignore[arg-type]
        )


def mae(model: RidgeModel, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(model.predict(x) - y)))

