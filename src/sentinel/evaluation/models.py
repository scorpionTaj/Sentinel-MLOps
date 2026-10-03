"""NumPy-only regressors compared in offline experiments.

All models share `fit(x, y) -> self` and `predict(x) -> np.ndarray`. Inputs are standardized
with training statistics inside each model so that nothing about validation data leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.model import RidgeModel


def _standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return x.mean(axis=0), scale


@dataclass
class MeanBaseline:
    """Predicts the training-set mean RUL: the floor any model must beat."""

    value: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> MeanBaseline:
        self.value = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)


@dataclass
class LinearRidge:
    """The serving model family (`sentinel.model.RidgeModel`)."""

    alpha: float = 1.0
    model: RidgeModel | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRidge:
        self.model = RidgeModel.fit(x, y, alpha=self.alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


@dataclass
class QuadraticRidge:
    """Ridge on standardized features plus their squares (captures the knee near the RUL cap)."""

    alpha: float = 1.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    model: RidgeModel | None = None

    def _expand(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.column_stack([z, z**2])

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuadraticRidge:
        self.mean, self.scale = _standardizer(x)
        self.model = RidgeModel.fit(self._expand(x), y, alpha=self.alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(self._expand(x))


@dataclass
class RandomFourierRidge:
    """Kernel ridge regression with an RBF kernel approximated by random Fourier features.

    gamma = gamma_scale / n_features so one grid works across feature sets of different width.
    The normal equations are accumulated in chunks to keep memory flat on 60k-row datasets.
    """

    gamma_scale: float = 1.0
    alpha: float = 1.0
    components: int = 600
    seed: int = 0
    chunk: int = 8192
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None
    offsets: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0

    def _features(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.sqrt(2.0 / self.components) * np.cos(z @ self.weights + self.offsets)

    def fit(self, x: np.ndarray, y: np.ndarray) -> RandomFourierRidge:
        self.mean, self.scale = _standardizer(x)
        rng = np.random.default_rng(self.seed)
        gamma = self.gamma_scale / x.shape[1]
        self.weights = rng.normal(0.0, np.sqrt(2 * gamma), size=(x.shape[1], self.components))
        self.offsets = rng.uniform(0.0, 2 * np.pi, size=self.components)
        self.intercept = float(np.mean(y))
        residual = y - self.intercept
        gram = np.zeros((self.components, self.components))
        moment = np.zeros(self.components)
        for start in range(0, len(x), self.chunk):
            phi = self._features(x[start : start + self.chunk])
            gram += phi.T @ phi
            moment += phi.T @ residual[start : start + self.chunk]
        gram[np.diag_indices_from(gram)] += self.alpha
        self.coefficients = np.linalg.solve(gram, moment)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x))
        for start in range(0, len(x), self.chunk):
            stop = start + self.chunk
            out[start:stop] = self.intercept + self._features(x[start:stop]) @ self.coefficients
        return out


MODELS = {
    "mean": MeanBaseline,
    "ridge": LinearRidge,
    "quadratic_ridge": QuadraticRidge,
    "rff_ridge": RandomFourierRidge,
}


def build_model(kind: str, params: dict[str, object], seed: int) -> object:
    if kind not in MODELS:
        raise ValueError(f"unknown model {kind!r}; choose from {sorted(MODELS)}")
    cls = MODELS[kind]
    if cls is RandomFourierRidge:
        return cls(**params, seed=seed)
    return cls(**params)
