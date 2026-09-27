"""Core Hilbert-space distortion diagnostics.

The implementation is deliberately finite-dimensional. A sampled EEG epoch is
treated as an element of R^(channels x samples), with the L2 inner product
approximated by a Riemann sum controlled by ``dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
FeatureMap = Callable[[Array], Array]


@dataclass(frozen=True)
class PairwiseIsometryReport:
    """Summary of pairwise metric distortion under a feature map."""

    n_pairs: int
    input_distance_mean: float
    output_distance_mean: float
    best_scale: float
    raw_abs_ratio_error_mean: float
    raw_abs_ratio_error_median: float
    scaled_abs_ratio_error_mean: float
    scaled_abs_ratio_error_median: float
    distance_spearman: float


@dataclass(frozen=True)
class BandOrthogonalityReport:
    """Summary of band-subspace orthogonality before and after a map."""

    band_names: tuple[str, ...]
    input_mean_abs_offdiag_cosine: float
    output_mean_abs_offdiag_cosine: float
    collapse_factor: float
    input_cosine_matrix: Array
    output_cosine_matrix: Array


@dataclass(frozen=True)
class AnchoredBandOrthogonalityReport:
    """Coherence of component responses after subtracting ``phi(0)``.

    This is distinct from :func:`band_orthogonality_report`, retained for
    compatibility with historical unanchored analyses. Components whose
    anchored norm is at most ``eps`` are excluded pairwise and counted.
    """

    band_names: tuple[str, ...]
    output_mean_abs_offdiag_cosine: float
    active_bands: int
    excluded_near_zero_bands: int
    zero_response_threshold: float
    output_cosine_matrix: Array


@dataclass(frozen=True)
class LocalDirectionalDistortionReport:
    """Finite-difference approximation to local metric distortion."""

    n_points: int
    n_directions: int
    eps: float
    stretch_mean: float
    stretch_median: float
    stretch_std: float
    stretch_min: float
    stretch_max: float


def flatten_epochs(x: Array) -> Array:
    """Return epochs as ``(n_epochs, n_features)``."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {x.shape}")
    return x.reshape(x.shape[0], -1)


def l2_norm(x: Array, dt: float = 1.0, axis: int | tuple[int, ...] | None = None) -> Array:
    """Discrete L2 norm with optional Riemann-sum time scaling."""

    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=axis) * dt)


def _sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> Array:
    if n < 2:
        raise ValueError("at least two samples are required")
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]
    return pairs


def _rankdata(a: Array) -> Array:
    """Average-rank transform with tie handling."""

    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(a: Array, b: Array) -> float:
    if len(a) < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom == 0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def pairwise_isometry_report(
    x: Array,
    phi: FeatureMap,
    *,
    dt: float = 1.0,
    n_pairs: int = 2048,
    seed: int = 0,
    eps: float = 1e-12,
) -> PairwiseIsometryReport:
    """Estimate pairwise distance distortion under ``phi``.

    Two versions are reported:
    - raw ratio error: ``|d_out / d_in - 1|``
    - scale-adjusted ratio error: same quantity after fitting the best global
      scalar ``s`` such that ``d_out ~= s * d_in``.

    The scale-adjusted score is the safer definition for EEG models because
    layer gains and unit conventions can change without destroying geometry.
    """

    x_flat = flatten_epochs(x)
    z_flat = flatten_epochs(phi(np.asarray(x)))
    if x_flat.shape[0] != z_flat.shape[0]:
        raise ValueError("feature map must preserve the number of epochs")

    rng = np.random.default_rng(seed)
    pairs = _sample_pairs(x_flat.shape[0], n_pairs, rng)
    dx = x_flat[pairs[:, 0]] - x_flat[pairs[:, 1]]
    dz = z_flat[pairs[:, 0]] - z_flat[pairs[:, 1]]
    d_in = l2_norm(dx, dt=dt, axis=1)
    d_out = l2_norm(dz, dt=1.0, axis=1)

    keep = d_in > eps
    d_in = d_in[keep]
    d_out = d_out[keep]
    ratios = d_out / np.maximum(d_in, eps)
    best_scale = float(np.dot(d_in, d_out) / max(np.dot(d_in, d_in), eps))
    scaled_ratios = d_out / np.maximum(best_scale * d_in, eps)

    return PairwiseIsometryReport(
        n_pairs=int(len(d_in)),
        input_distance_mean=float(np.mean(d_in)),
        output_distance_mean=float(np.mean(d_out)),
        best_scale=best_scale,
        raw_abs_ratio_error_mean=float(np.mean(np.abs(ratios - 1.0))),
        raw_abs_ratio_error_median=float(np.median(np.abs(ratios - 1.0))),
        scaled_abs_ratio_error_mean=float(np.mean(np.abs(scaled_ratios - 1.0))),
        scaled_abs_ratio_error_median=float(np.median(np.abs(scaled_ratios - 1.0))),
        distance_spearman=_spearman(d_in, d_out),
    )


def cosine_matrix(vectors: Array, eps: float = 1e-12) -> Array:
    vectors = np.asarray(vectors, dtype=np.float64)
    flat = vectors.reshape(vectors.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norms, eps)
    return flat @ flat.T


def offdiag_mean_abs(matrix: Array) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def band_orthogonality_report(
    band_components: Array,
    phi: FeatureMap,
    *,
    band_names: tuple[str, ...] | None = None,
) -> BandOrthogonalityReport:
    """Measure whether a feature map preserves band-component angles.

    ``band_components`` must have shape ``(n_bands, channels, samples)`` for a
    single epoch, or any shape where the first axis indexes bands.
    """

    band_components = np.asarray(band_components, dtype=np.float64)
    n_bands = band_components.shape[0]
    if band_names is None:
        band_names = tuple(f"band_{i}" for i in range(n_bands))
    if len(band_names) != n_bands:
        raise ValueError("band_names length must match number of bands")

    features = phi(band_components)
    input_cos = cosine_matrix(band_components)
    output_cos = cosine_matrix(features)
    input_odi = offdiag_mean_abs(input_cos)
    output_odi = offdiag_mean_abs(output_cos)
    collapse_factor = output_odi / max(input_odi, 1e-12)
    return BandOrthogonalityReport(
        band_names=tuple(band_names),
        input_mean_abs_offdiag_cosine=input_odi,
        output_mean_abs_offdiag_cosine=output_odi,
        collapse_factor=float(collapse_factor),
        input_cosine_matrix=input_cos,
        output_cosine_matrix=output_cos,
    )


def anchored_band_orthogonality_report(
    band_components: Array,
    phi: FeatureMap,
    *,
    band_names: tuple[str, ...] | None = None,
    eps: float = 1e-12,
) -> AnchoredBandOrthogonalityReport:
    """Compute anchored component coherence with an explicit degeneracy rule."""
    components = np.asarray(band_components, dtype=np.float64)
    n_bands = components.shape[0]
    names = tuple(f"band_{i}" for i in range(n_bands)) if band_names is None else tuple(band_names)
    if len(names) != n_bands:
        raise ValueError("band_names length must match number of bands")
    responses = np.asarray(phi(components), dtype=np.float64)
    zero_input = np.zeros_like(components[:1])
    baseline = np.asarray(phi(zero_input), dtype=np.float64)[0]
    anchored = responses - baseline
    flat = anchored.reshape(n_bands, -1)
    norms = np.linalg.norm(flat, axis=1)
    active = norms > eps
    cosine = np.full((n_bands, n_bands), np.nan, dtype=np.float64)
    if active.any():
        unit = flat[active] / norms[active, None]
        cosine[np.ix_(active, active)] = unit @ unit.T
    active_count = int(active.sum())
    if active_count < 2:
        odi = float("nan")
    else:
        block = cosine[np.ix_(active, active)]
        odi = float(np.mean(np.abs(block[~np.eye(active_count, dtype=bool)])))
    return AnchoredBandOrthogonalityReport(
        band_names=names,
        output_mean_abs_offdiag_cosine=odi,
        active_bands=active_count,
        excluded_near_zero_bands=int((~active).sum()),
        zero_response_threshold=float(eps),
        output_cosine_matrix=cosine,
    )


def local_directional_distortion(
    x: Array,
    phi: FeatureMap,
    *,
    dt: float = 1.0,
    n_points: int = 16,
    n_directions: int = 8,
    eps: float = 1e-3,
    seed: int = 0,
) -> LocalDirectionalDistortionReport:
    """Approximate local stretch factors with random finite directions."""

    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    point_idx = rng.choice(n, size=min(n_points, n), replace=False)
    stretches: list[float] = []

    for idx in point_idx:
        x0 = x[idx : idx + 1]
        y0 = flatten_epochs(phi(x0))[0]
        for _ in range(n_directions):
            direction = rng.normal(size=x0.shape)
            direction_norm = l2_norm(direction, dt=dt)
            if direction_norm == 0:
                continue
            direction = direction / direction_norm
            y1 = flatten_epochs(phi(x0 + eps * direction))[0]
            stretch = np.linalg.norm(y1 - y0) / eps
            stretches.append(float(stretch))

    stretches_arr = np.asarray(stretches, dtype=np.float64)
    return LocalDirectionalDistortionReport(
        n_points=int(len(point_idx)),
        n_directions=int(n_directions),
        eps=float(eps),
        stretch_mean=float(np.mean(stretches_arr)),
        stretch_median=float(np.median(stretches_arr)),
        stretch_std=float(np.std(stretches_arr)),
        stretch_min=float(np.min(stretches_arr)),
        stretch_max=float(np.max(stretches_arr)),
    )
