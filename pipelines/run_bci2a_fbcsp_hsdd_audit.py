#!/usr/bin/env python3
"""Audit filter-bank CSP feature geometry on BCI IV 2a."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from scipy.signal import sosfiltfilt, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsdd.diagnostics import _spearman, cosine_matrix, l2_norm, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from train_eegnet_bci2a_layer_audit import (
    label_counts,
    load_bci2a,
    make_train_val,
    set_seed,
)


DEFAULT_BANDS = "4-8,8-12,12-16,16-20,20-24,24-28,28-32,32-36,36-40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--csp-components", type=int, default=4)
    parser.add_argument("--n-pairs", type=int, default=4096)
    parser.add_argument("--band-epochs", type=int, default=180)
    parser.add_argument("--local-points", type=int, default=12)
    parser.add_argument("--local-directions", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-3)
    return parser.parse_args()


def parse_bands(spec: str) -> list[tuple[float, float]]:
    bands = []
    for item in spec.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split("-")
        bands.append((float(lo), float(hi)))
    if not bands:
        raise ValueError("--bands must contain at least one low-high pair")
    return bands


def band_name(band: tuple[float, float]) -> str:
    lo, hi = band
    return f"{lo:g}_{hi:g}hz"


def make_sos(band: tuple[float, float], sfreq: float) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * sfreq
    hi = min(float(hi), nyq - 1e-3)
    lo = max(float(lo), 0.5)
    if not lo < hi:
        raise ValueError(f"invalid band {band} for sfreq={sfreq}")
    return butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")


def apply_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def build_csp(n_components: int):
    from mne.decoding import CSP

    try:
        return CSP(
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            transform_into="average_power",
            norm_trace=False,
        )
    except TypeError:
        return CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)


class FBCSPMap:
    def __init__(
        self,
        *,
        bands: list[tuple[float, float]],
        sfreq: float,
        csp_components: int,
    ) -> None:
        self.bands = bands
        self.sfreq = sfreq
        self.csp_components = csp_components
        self.sos = [make_sos(band, sfreq) for band in bands]
        self.csps = [build_csp(csp_components) for _ in bands]
        self.classifier = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        self.band_layer_names = tuple(f"csp_{band_name(band)}" for band in bands)
        self.layer_names = self.band_layer_names + ("fbcsp_concat", "lda_logits")

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        per_band = []
        for sos, csp in zip(self.sos, self.csps):
            xb = apply_filter(train_x, sos)
            try:
                csp.fit(xb, train_y)
            except Exception:
                csp.reg = None
                csp.fit(xb, train_y)
            per_band.append(csp.transform(xb).astype(np.float32))
        features = np.concatenate(per_band, axis=1)
        self.classifier.fit(features, train_y)

    def transform_layers(self, x: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        per_band = []
        for band, sos, csp in zip(self.bands, self.sos, self.csps):
            xb = apply_filter(np.asarray(x, dtype=np.float32), sos)
            z = csp.transform(xb).astype(np.float32, copy=False)
            out[f"csp_{band_name(band)}"] = z
            per_band.append(z)
        features = np.concatenate(per_band, axis=1).astype(np.float32, copy=False)
        out["fbcsp_concat"] = features
        out["lda_logits"] = self.classifier.decision_function(features).astype(np.float32, copy=False)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.transform_layers(x)["fbcsp_concat"])

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean(pred == y))


def pairwise_isometry(
    x: np.ndarray,
    layers: dict[str, np.ndarray],
    *,
    dt: float,
    n_pairs: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]
    d_in = l2_norm(
        x[pairs[:, 0]].reshape(n_pairs, -1) - x[pairs[:, 1]].reshape(n_pairs, -1),
        dt=dt,
        axis=1,
    )
    out = {}
    for name, z_raw in layers.items():
        z = z_raw.reshape(z_raw.shape[0], -1).astype(np.float64)
        d_out = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
        best_scale = float(np.dot(d_in, d_out) / max(np.dot(d_in, d_in), 1e-12))
        raw_ratios = d_out / np.maximum(d_in, 1e-12)
        scaled_ratios = d_out / np.maximum(best_scale * d_in, 1e-12)
        out[name] = {
            "n_pairs": int(n_pairs),
            "best_scale": best_scale,
            "input_distance_mean": float(np.mean(d_in)),
            "output_distance_mean": float(np.mean(d_out)),
            "raw_abs_ratio_error_mean": float(np.mean(np.abs(raw_ratios - 1.0))),
            "scaled_abs_ratio_error_mean": float(np.mean(np.abs(scaled_ratios - 1.0))),
            "scaled_abs_ratio_error_median": float(np.median(np.abs(scaled_ratios - 1.0))),
            "distance_spearman": _spearman(d_in, d_out),
        }
    return out


def band_orthogonality(
    x: np.ndarray,
    fmap: FBCSPMap,
    *,
    sfreq: float,
    n_epochs: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_epochs, len(x)), replace=False)
    per_layer = {
        name: {"input": [], "output": [], "input_matrix": [], "output_matrix": []}
        for name in fmap.layer_names
    }
    band_names = None
    for idx in indices:
        components, names = fft_band_components(x[idx], sfreq=sfreq)
        band_names = names
        input_cos = cosine_matrix(components)
        layers = fmap.transform_layers(components.astype(np.float32))
        for name in fmap.layer_names:
            output_cos = cosine_matrix(layers[name])
            per_layer[name]["input"].append(offdiag_mean_abs(input_cos))
            per_layer[name]["output"].append(offdiag_mean_abs(output_cos))
            per_layer[name]["input_matrix"].append(input_cos)
            per_layer[name]["output_matrix"].append(output_cos)
    reports = {}
    for name, values in per_layer.items():
        reports[name] = {
            "band_names": band_names,
            "n_epochs": int(len(indices)),
            "input_mean_abs_offdiag_cosine_mean": float(np.mean(values["input"])),
            "output_mean_abs_offdiag_cosine_mean": float(np.mean(values["output"])),
            "output_mean_abs_offdiag_cosine_std": float(np.std(values["output"])),
            "collapse_factor_mean": float(
                np.mean(values["output"]) / max(float(np.mean(values["input"])), 1e-12)
            ),
            "input_cosine_matrix_mean": np.mean(values["input_matrix"], axis=0),
            "output_cosine_matrix_mean": np.mean(values["output_matrix"], axis=0),
        }
    return reports


def local_directional_distortion(
    x: np.ndarray,
    fmap: FBCSPMap,
    *,
    dt: float,
    n_points: int,
    n_directions: int,
    eps: float,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_points, len(x)), replace=False)
    stretches = {name: [] for name in fmap.layer_names}
    for idx in indices:
        x0 = x[idx : idx + 1].astype(np.float32)
        base = fmap.transform_layers(x0)
        directions = rng.normal(size=(n_directions,) + x0.shape[1:]).astype(np.float32)
        norms = l2_norm(directions.reshape(n_directions, -1), dt=dt, axis=1)
        directions = directions / np.maximum(norms[:, None, None], 1e-12)
        for direction in directions:
            perturbed = fmap.transform_layers((x0 + eps * direction[None]).astype(np.float32))
            for name in fmap.layer_names:
                z0 = base[name].reshape(1, -1)
                z1 = perturbed[name].reshape(1, -1)
                stretches[name].append(float(np.linalg.norm(z1 - z0) / eps))
    reports = {}
    for name, vals in stretches.items():
        arr = np.asarray(vals, dtype=np.float64)
        reports[name] = {
            "n_points": int(len(indices)),
            "n_directions": int(n_directions),
            "eps": float(eps),
            "stretch_mean": float(np.mean(arr)),
            "stretch_median": float(np.median(arr)),
            "stretch_std": float(np.std(arr)),
            "stretch_min": float(np.min(arr)),
            "stretch_max": float(np.max(arr)),
        }
    return reports


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    bands = parse_bands(args.bands)
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
    sfreq = 1.0 / data["dt"]
    fmap = FBCSPMap(bands=bands, sfreq=sfreq, csp_components=args.csp_components)
    fmap.fit(train_x, train_y)
    train_acc = fmap.score(train_x, train_y)
    val_acc = fmap.score(val_x, val_y)
    print(f"FBCSP label_mode={args.label_mode} seed={args.seed} train_acc={train_acc:.3f} val_acc={val_acc:.3f}", flush=True)

    audit_layers = fmap.transform_layers(data["x_all"])
    pairwise = pairwise_isometry(data["x_all"], audit_layers, dt=data["dt"], n_pairs=args.n_pairs, seed=args.seed)
    band = band_orthogonality(data["x_all"], fmap, sfreq=sfreq, n_epochs=args.band_epochs, seed=args.seed)
    local = local_directional_distortion(
        data["x_all"],
        fmap,
        dt=data["dt"],
        n_points=args.local_points,
        n_directions=args.local_directions,
        eps=args.eps,
        seed=args.seed,
    )

    layers = {
        name: {
            "pairwise_isometry": pairwise[name],
            "band_orthogonality": band[name],
            "local_directional_distortion": local[name],
        }
        for name in fmap.layer_names
    }
    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "labels_all": label_counts(data["labels_all"]),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "train_label_counts": {str(i): int((train_y == i).sum()) for i in sorted(set(train_y))},
            "val_label_counts": {str(i): int((val_y == i).sum()) for i in sorted(set(val_y))},
            "dt": data["dt"],
            "sfreq": sfreq,
        },
        "model": {
            "arch": "fbcsp",
            "source": "mne.decoding.CSP filter bank + sklearn LinearDiscriminantAnalysis",
            "bands": [[lo, hi] for lo, hi in bands],
            "csp_components_per_band": args.csp_components,
            "layer_names": list(fmap.layer_names),
        },
        "training": {
            "label_mode": args.label_mode,
            "seed": args.seed,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_val_acc": val_acc,
            "final_val_acc": val_acc,
            "final_train_acc": train_acc,
        },
        "layers": layers,
    }
    out = Path(args.out or f"reports/bci2a_fbcsp_hsdd_{args.label_mode}_seed{args.seed}.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
