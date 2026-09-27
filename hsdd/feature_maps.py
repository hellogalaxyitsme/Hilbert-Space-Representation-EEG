"""Simple feature maps for feasibility controls."""

from __future__ import annotations

import numpy as np


class IdentityMap:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)


class LinearProjection:
    """Random linear projection with optional orthonormal rows."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        seed: int = 0,
        orthonormal_rows: bool = True,
    ) -> None:
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(output_dim, input_dim))
        if orthonormal_rows:
            q, _ = np.linalg.qr(weights.T)
            weights = q[:, :output_dim].T
        else:
            weights = weights / np.sqrt(input_dim)
        self.weights = weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(x.shape[0], -1)
        return flat @ self.weights.T


class ReluProjection:
    """Random projection followed by ReLU."""

    def __init__(self, input_dim: int, output_dim: int, *, seed: int = 0) -> None:
        self.projection = LinearProjection(
            input_dim,
            output_dim,
            seed=seed,
            orthonormal_rows=False,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.projection(x), 0.0)


class BandMixingMap:
    """A deterministic map that intentionally mixes orthogonal components."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(x.shape[0], -1)
        shared = np.mean(flat, axis=1, keepdims=True)
        energy = np.sqrt(np.mean(flat**2, axis=1, keepdims=True))
        return np.concatenate([shared, energy, shared + energy], axis=1)
