#!/usr/bin/env python3
"""Audit classical filter-bank log-variance EEG features on active datasets."""

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
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsdd.diagnostics import _spearman, cosine_matrix, l2_norm, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from p300_utils import label_counts as p300_label_counts
from p300_utils import load_p300_cache
from seediv_utils import label_counts as seediv_label_counts
from seediv_utils import load_seediv_cache
from train_eegnet_bci2a_layer_audit import load_bci2a, make_train_val, set_seed
from train_sleepedf_arch_layer_audit import label_counts as sleep_label_counts
from train_sleepedf_arch_layer_audit import load_cache as load_sleepedf_cache


DEFAULT_BANDS = "1-4,4-8,8-13,13-30,30-45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bci2a", "sleepedf_full", "seediv", "p300"), required=True)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--n-pairs", type=int, default=4096)
    parser.add_argument("--band-epochs", type=int, default=180)
    parser.add_argument("--local-points", type=int, default=12)
    parser.add_argument("--local-directions", type=int, default=4)
    parser.add_argument("--max-audit-epochs", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--bci-root", default=None)
    parser.add_argument("--sleep-cache", default="prepared/sleepedf_sc153_fpzcz_pzoz.npz")
    parser.add_argument("--sleep-val-recordings", type=int, default=31)
    parser.add_argument("--sleep-max-train-per-class", type=int, default=1600)
    parser.add_argument("--sleep-max-val-per-class", type=int, default=400)
    parser.add_argument("--seediv-cache", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--seediv-val-subjects", default="13,14,15")
    parser.add_argument("--seediv-max-train-per-class", type=int, default=800)
    parser.add_argument("--seediv-max-val-per-class", type=int, default=200)
    parser.add_argument("--p300-cache", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--p300-val-subjects", default="9,10")
    parser.add_argument("--p300-max-train-per-class", type=int, default=288)
    parser.add_argument("--p300-max-val-per-class", type=int, default=72)
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
    return f"{band[0]:g}_{band[1]:g}hz"


def make_sos(band: tuple[float, float], sfreq: float) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * sfreq
    lo = max(float(lo), 0.5)
    hi = min(float(hi), nyq - 1e-3)
    if hi <= lo:
        raise ValueError(f"invalid band {band} for sfreq={sfreq}")
    return butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")


def apply_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci2a":
        data = load_bci2a(Path(args.bci_root), tmin=0.0, tmax=3.0)
        train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
        return {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "x": data["x_all"].astype(np.float32),
            "y": data["labels_all"].astype(np.int64),
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "sfreq": 1.0 / float(data["dt"]),
            "dt": float(data["dt"]),
            "label_counts": {str(k): int(v) for k, v in data["labels_all_counts"].items()} if "labels_all_counts" in data else {},
            "label_names": ["left", "right", "feet", "tongue"],
            "split": {"heldout": "A09T original validation split"},
        }
    if args.dataset == "sleepedf_full":
        data = load_sleepedf_cache(
            Path(args.sleep_cache),
            args.sleep_val_recordings,
            args.seed,
            args.sleep_max_train_per_class,
            args.sleep_max_val_per_class,
        )
        train_y = data["train_y"].copy()
        if args.label_mode == "shuffled":
            train_y = np.random.default_rng(args.seed).permutation(train_y)
        return {
            "kind": "Sleep_EDF_full_sleep_cassette_cache",
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": float(data["sfreq"]),
            "dt": 1.0 / float(data["sfreq"]),
            "label_counts": sleep_label_counts(data["y"], data["label_names"]),
            "label_names": data["label_names"],
            "split": {"val_recordings": data["val_recordings"]},
        }
    if args.dataset == "seediv":
        data = load_seediv_cache(
            Path(args.seediv_cache),
            val_subjects=args.seediv_val_subjects,
            seed=args.seed,
            max_train_per_class=args.seediv_max_train_per_class,
            max_val_per_class=args.seediv_max_val_per_class,
        )
        train_y = data["train_y"].copy()
        if args.label_mode == "shuffled":
            train_y = np.random.default_rng(args.seed).permutation(train_y)
        return {
            "kind": "SEED_IV_raw_eeg_window_cache",
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": float(data["sfreq"]),
            "dt": float(data["dt"]),
            "label_counts": seediv_label_counts(data["y"], data["label_names"]),
            "label_names": data["label_names"],
            "split": {"train_subjects": data["train_subjects"], "val_subjects": data["val_subjects"]},
        }
    data = load_p300_cache(
        Path(args.p300_cache),
        val_subjects=args.p300_val_subjects,
        seed=args.seed,
        max_train_per_class=args.p300_max_train_per_class,
        max_val_per_class=args.p300_max_val_per_class,
    )
    train_y = data["train_y"].copy()
    if args.label_mode == "shuffled":
        train_y = np.random.default_rng(args.seed).permutation(train_y)
    return {
        "kind": "BNCI2014_009_P300_raw_event_cache",
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "train_x": data["train_x"],
        "train_y": train_y,
        "val_x": data["val_x"],
        "val_y": data["val_y"],
        "sfreq": float(data["sfreq"]),
        "dt": float(data["dt"]),
        "label_counts": p300_label_counts(data["y"], data["label_names"]),
        "label_names": data["label_names"],
        "split": {"train_subjects": data["train_subjects"], "val_subjects": data["val_subjects"]},
    }


class FilterBankLogVarMap:
    def __init__(self, *, bands: list[tuple[float, float]], sfreq: float) -> None:
        self.bands = bands
        self.sfreq = sfreq
        self.sos = [make_sos(band, sfreq) for band in bands]
        self.classifier = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        self.band_layer_names = tuple(f"logvar_{band_name(band)}" for band in bands)
        self.layer_names = self.band_layer_names + ("filterbank_concat", "lda_logits")

    def transform_features(self, x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        layers = {}
        parts = []
        for band, sos in zip(self.bands, self.sos):
            xb = apply_filter(np.asarray(x, dtype=np.float32), sos)
            # Log-variance per channel is a classical band-power representation.
            z = np.log(np.var(xb, axis=-1) + 1e-12).astype(np.float32, copy=False)
            layers[f"logvar_{band_name(band)}"] = z
            parts.append(z)
        concat = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        layers["filterbank_concat"] = concat
        return layers, concat

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        _layers, features = self.transform_features(train_x)
        self.classifier.fit(features, train_y)

    def transform_layers(self, x: np.ndarray) -> dict[str, np.ndarray]:
        layers, features = self.transform_features(x)
        layers["lda_logits"] = self.classifier.decision_function(features).astype(np.float32, copy=False)
        return layers

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        _layers, features = self.transform_features(x)
        return float(np.mean(self.classifier.predict(features) == y))


def pairwise_isometry(x: np.ndarray, layers: dict[str, np.ndarray], *, dt: float, n_pairs: int, seed: int) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]
    d_in = l2_norm(x[pairs[:, 0]].reshape(n_pairs, -1) - x[pairs[:, 1]].reshape(n_pairs, -1), dt=dt, axis=1)
    out = {}
    for name, z_raw in layers.items():
        z = z_raw.reshape(z_raw.shape[0], -1).astype(np.float64)
        d_out = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
        best_scale = float(np.dot(d_in, d_out) / max(np.dot(d_in, d_in), 1e-12))
        scaled_ratios = d_out / np.maximum(best_scale * d_in, 1e-12)
        out[name] = {
            "n_pairs": int(n_pairs),
            "best_scale": best_scale,
            "input_distance_mean": float(np.mean(d_in)),
            "output_distance_mean": float(np.mean(d_out)),
            "scaled_abs_ratio_error_mean": float(np.mean(np.abs(scaled_ratios - 1.0))),
            "scaled_abs_ratio_error_median": float(np.median(np.abs(scaled_ratios - 1.0))),
            "distance_spearman": _spearman(d_in, d_out),
        }
    return out


def band_orthogonality(x: np.ndarray, fmap: FilterBankLogVarMap, *, sfreq: float, n_epochs: int, seed: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_epochs, len(x)), replace=False)
    per_layer = {name: {"input": [], "output": []} for name in fmap.layer_names}
    band_names = None
    for idx in indices:
        components, names = fft_band_components(x[idx], sfreq=sfreq)
        band_names = names
        input_odi = offdiag_mean_abs(cosine_matrix(components))
        layers = fmap.transform_layers(components.astype(np.float32))
        for name in fmap.layer_names:
            output_odi = offdiag_mean_abs(cosine_matrix(layers[name]))
            per_layer[name]["input"].append(input_odi)
            per_layer[name]["output"].append(output_odi)
    reports = {}
    for name, vals in per_layer.items():
        in_mean = float(np.mean(vals["input"]))
        out_mean = float(np.mean(vals["output"]))
        reports[name] = {
            "band_names": band_names,
            "n_epochs": int(len(indices)),
            "input_mean_abs_offdiag_cosine_mean": in_mean,
            "output_mean_abs_offdiag_cosine_mean": out_mean,
            "output_mean_abs_offdiag_cosine_std": float(np.std(vals["output"])),
            "collapse_factor_mean": out_mean / max(in_mean, 1e-12),
        }
    return reports


def local_directional_distortion(
    x: np.ndarray,
    fmap: FilterBankLogVarMap,
    *,
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
        flat = directions.reshape(n_directions, -1)
        directions = directions / np.maximum(np.linalg.norm(flat, axis=1)[:, None, None], 1e-12)
        for direction in directions:
            perturbed = fmap.transform_layers((x0 + eps * direction[None]).astype(np.float32))
            for name in fmap.layer_names:
                stretches[name].append(float(np.linalg.norm(perturbed[name].reshape(1, -1) - base[name].reshape(1, -1)) / eps))
    return {
        name: {
            "n_points": int(len(indices)),
            "n_directions": int(n_directions),
            "eps": float(eps),
            "stretch_mean": float(np.mean(vals)),
            "stretch_median": float(np.median(vals)),
            "stretch_std": float(np.std(vals)),
        }
        for name, vals in stretches.items()
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    bands = parse_bands(args.bands)
    data = load_dataset(args)
    fmap = FilterBankLogVarMap(bands=bands, sfreq=data["sfreq"])
    fmap.fit(data["train_x"], data["train_y"])
    train_acc = fmap.score(data["train_x"], data["train_y"])
    val_acc = fmap.score(data["val_x"], data["val_y"])
    print(f"dataset={args.dataset} mode={args.label_mode} seed={args.seed} train_acc={train_acc:.3f} val_acc={val_acc:.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    audit_size = min(args.max_audit_epochs, len(data["x"]))
    audit_idx = rng.choice(len(data["x"]), size=audit_size, replace=False)
    audit_x = data["x"][audit_idx].astype(np.float32, copy=False)
    audit_layers = fmap.transform_layers(audit_x)
    pairwise = pairwise_isometry(audit_x, audit_layers, dt=data["dt"], n_pairs=args.n_pairs, seed=args.seed)
    band = band_orthogonality(audit_x, fmap, sfreq=data["sfreq"], n_epochs=args.band_epochs, seed=args.seed)
    local = local_directional_distortion(
        audit_x,
        fmap,
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
            "name": args.dataset,
            "kind": data["kind"],
            "x_shape": list(data["x"].shape),
            "audit_shape": list(audit_x.shape),
            "audit_indices": audit_idx.astype(int).tolist(),
            "train_shape": list(data["train_x"].shape),
            "val_shape": list(data["val_x"].shape),
            "sfreq": data["sfreq"],
            "dt": data["dt"],
            "label_names": data["label_names"],
            "label_counts": data["label_counts"],
            "split": data["split"],
        },
        "model": {
            "arch": "filterbank_logvar_lda",
            "source": "Butterworth filter bank + per-channel log variance + shrinkage LDA",
            "bands": [[lo, hi] for lo, hi in bands],
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
    out = Path(args.out or f"reports/{args.dataset}_filterbank_logvar_hsdd_{args.label_mode}_seed{args.seed}.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
