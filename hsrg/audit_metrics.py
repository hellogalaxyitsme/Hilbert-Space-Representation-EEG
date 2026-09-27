"""Numerically explicit raw and zero-anchored band-coherence estimators."""

from __future__ import annotations

import numpy as np


def paired_band_metrics(
    responses: np.ndarray,
    baseline: np.ndarray,
    *,
    eps: float = 1e-12,
) -> list[dict[str, float | int]]:
    """Return per-window raw and anchored estimates and signed decomposition.

    ``responses`` has shape ``(windows, bands, features)``. Raw coherence
    uses every off-diagonal band pair and the product of individually floored
    response norms. The baseline, cross, and residual terms reconstruct the
    signed raw cosine; they are not components of absolute ODI.
    """
    z = np.asarray(responses, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if z.ndim != 3 or z.shape[-1] != b.size:
        raise ValueError("responses must be (windows, bands, baseline.size)")
    if z.shape[0] == 0 or z.shape[1] < 2 or z.shape[2] == 0:
        raise ValueError("responses require at least one window, two bands, and one feature")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(z).all() or not np.isfinite(b).all():
        raise ValueError("responses and baseline must be finite and eps positive")
    raw_norm = np.linalg.norm(z, axis=-1)
    residual = z - b[None, None, :]
    residual_norm = np.linalg.norm(residual, axis=-1)
    denominator = np.maximum(raw_norm[:, :, None], eps) * np.maximum(raw_norm[:, None, :], eps)
    raw_signed = np.einsum("wkd,wjd->wkj", z, z) / denominator
    baseline_sq = float(np.dot(b, b))
    baseline_signed = baseline_sq / denominator
    cross = np.einsum("d,wkd->wk", b, residual)
    cross_signed = (cross[:, :, None] + cross[:, None, :]) / denominator
    residual_signed = np.einsum("wkd,wjd->wkj", residual, residual) / denominator
    mask = ~np.eye(z.shape[1], dtype=bool)
    results: list[dict[str, float | int]] = []
    for index in range(z.shape[0]):
        active = residual_norm[index] > eps
        anchored_mask = np.outer(active, active) & mask
        anchored = float("nan")
        anchor_signed = float("nan")
        if anchored_mask.any():
            unit = residual[index] / np.maximum(residual_norm[index, :, None], eps)
            anchored_cos = unit @ unit.T
            anchored = float(np.abs(anchored_cos[anchored_mask]).mean())
            anchor_signed = float(anchored_cos[anchored_mask].mean())
        reconstructed = baseline_signed[index] + cross_signed[index] + residual_signed[index]
        component_scale = (
            np.abs(baseline_signed[index])
            + np.abs(cross_signed[index])
            + np.abs(residual_signed[index])
        )
        results.append({
            "raw_odi": float(np.abs(raw_signed[index][mask]).mean()),
            "anchored_odi": anchored,
            "raw_signed_mean": float(raw_signed[index][mask].mean()),
            "anchor_signed_mean": anchor_signed,
            "active_bands": int(active.sum()),
            "excluded_bands": int((~active).sum()),
            "raw_floor_bands": int((raw_norm[index] < eps).sum()),
            "raw_floor_pair_count": int(((raw_norm[index, :, None] < eps) | (raw_norm[index, None, :] < eps))[mask].sum()),
            "anchor_excluded_pair_count": int(mask.sum() - anchored_mask.sum()),
            "baseline_sq_over_raw_den": float(baseline_signed[index][mask].mean()),
            "cross_over_raw_den": float(cross_signed[index][mask].mean()),
            "residual_over_raw_den": float(residual_signed[index][mask].mean()),
            "decomposition_max_abs_error": float(np.abs(reconstructed - raw_signed[index]).max()),
            "decomposition_error_bound": float((32 * np.finfo(np.float64).eps * component_scale[mask]).max()),
        })
    return results
