#!/usr/bin/env python3
"""HSDD hyperparameter sensitivity on trained BCI IV 2a deep models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from hsdd.diagnostics import _spearman, cosine_matrix, l2_norm, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import LayerCapture, load_bci2a, make_train_val, set_seed


CANONICAL_CLINICAL = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 45.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--max-epochs", type=int, default=384)
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--n-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pair-metrics", default="euclidean,cosine,correlation,l1")
    parser.add_argument("--band-schemes", default="canonical,narrow_2hz,data_driven_equal_power")
    return parser.parse_args()


def parse_floats(spec: str) -> list[float]:
    return [float(item) for item in spec.replace(",", " ").split() if item.strip()]


def parse_items(spec: str) -> list[str]:
    return [item.strip() for item in spec.replace(",", " ").split() if item.strip()]


def checkpoint_path(arch: str, mode: str, seed: int) -> Path:
    if seed == 41:
        candidates = [
            Path(f"reports/{arch}_bci2a_{mode}.pt"),
            Path(f"reports/{arch}_bci2a.pt") if mode == "true" else Path("__missing__"),
        ]
    else:
        candidates = [Path(f"reports/{arch}_bci2a_{mode}_seed{seed}.pt")]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing checkpoint for arch={arch} mode={mode} seed={seed}: {candidates}")


def narrow_2hz_bands(low: float = 4.0, high: float = 40.0) -> tuple[tuple[str, float, float], ...]:
    bands = []
    start = low
    while start < high - 1e-9:
        stop = min(start + 2.0, high)
        bands.append((f"{start:g}_{stop:g}hz", start, stop))
        start = stop
    return tuple(bands)


def data_driven_equal_power_bands(x: np.ndarray, sfreq: float, *, low: float = 1.0, high: float = 45.0, n_bands: int = 5):
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    spectrum = np.fft.rfft(x.astype(np.float64), axis=-1)
    power = np.mean(np.abs(spectrum) ** 2, axis=(0, 1))
    mask = (freqs >= low) & (freqs <= high)
    freq_sel = freqs[mask]
    power_sel = power[mask]
    if len(freq_sel) < n_bands + 1 or float(power_sel.sum()) <= 0.0:
        return CANONICAL_CLINICAL
    cdf = np.cumsum(power_sel)
    cdf = cdf / cdf[-1]
    boundaries = [low]
    for q in np.linspace(0.0, 1.0, n_bands + 1)[1:-1]:
        idx = int(np.searchsorted(cdf, q, side="left"))
        boundaries.append(float(freq_sel[min(idx, len(freq_sel) - 1)]))
    boundaries.append(high)
    cleaned = [float(boundaries[0])]
    for value in boundaries[1:]:
        cleaned.append(max(float(value), cleaned[-1] + 0.5))
    cleaned[-1] = max(cleaned[-1], cleaned[-2] + 0.5)
    cleaned[-1] = min(cleaned[-1], high)
    bands = []
    for idx, (lo, hi) in enumerate(zip(cleaned[:-1], cleaned[1:])):
        if hi <= lo:
            hi = lo + 0.5
        bands.append((f"dd_power_{idx}_{lo:g}_{hi:g}hz", lo, hi))
    return tuple(bands)


def band_schemes(names: list[str], x: np.ndarray, sfreq: float) -> dict[str, tuple[tuple[str, float, float], ...]]:
    out = {}
    for name in names:
        if name == "canonical":
            out[name] = CANONICAL_CLINICAL
        elif name == "narrow_2hz":
            out[name] = narrow_2hz_bands()
        elif name == "data_driven_equal_power":
            out[name] = data_driven_equal_power_bands(x, sfreq)
        else:
            raise ValueError(f"unknown band scheme: {name}")
    return out


def capture_activations(model, x: np.ndarray, layer_names: tuple[str, ...], *, device: torch.device, batch_size: int):
    capture = LayerCapture(model.eval(), layer_names)
    chunks = {name: [] for name in layer_names}
    try:
        for start in range(0, len(x), batch_size):
            outputs = capture(x[start : start + batch_size], device=device)
            for name in layer_names:
                chunks[name].append(outputs[name].reshape(outputs[name].shape[0], -1).astype(np.float32))
    finally:
        capture.close()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def distance(values_a: np.ndarray, values_b: np.ndarray, metric: str, *, dt: float, is_input: bool) -> np.ndarray:
    if metric == "euclidean":
        scale_dt = dt if is_input else 1.0
        return l2_norm(values_a - values_b, dt=scale_dt, axis=1)
    if metric == "l1":
        scale = dt if is_input else 1.0
        return np.sum(np.abs(values_a - values_b), axis=1) * scale
    if metric in {"cosine", "correlation"}:
        a = values_a.astype(np.float64)
        b = values_b.astype(np.float64)
        if metric == "correlation":
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        sim = np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)
        return 1.0 - sim
    raise ValueError(metric)


def pairwise_report(
    x_flat: np.ndarray,
    z_flat: np.ndarray,
    *,
    metric: str,
    dt: float,
    n_pairs: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    n = len(x_flat)
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]
    d_in = distance(x_flat[pairs[:, 0]], x_flat[pairs[:, 1]], metric, dt=dt, is_input=True)
    d_out = distance(z_flat[pairs[:, 0]], z_flat[pairs[:, 1]], metric, dt=dt, is_input=False)
    keep = d_in > 1e-12
    d_in = d_in[keep]
    d_out = d_out[keep]
    best_scale = float(np.dot(d_in, d_out) / max(np.dot(d_in, d_in), 1e-12))
    scaled = d_out / np.maximum(best_scale * d_in, 1e-12)
    return {
        "distance_metric": metric,
        "n_pairs": int(len(d_in)),
        "best_scale": best_scale,
        "input_distance_mean": float(np.mean(d_in)),
        "output_distance_mean": float(np.mean(d_out)),
        "scaled_abs_ratio_error_mean": float(np.mean(np.abs(scaled - 1.0))),
        "scaled_abs_ratio_error_median": float(np.median(np.abs(scaled - 1.0))),
        "distance_spearman": _spearman(d_in, d_out),
    }


def band_odi_for_subset(
    model,
    x: np.ndarray,
    layer_names: tuple[str, ...],
    *,
    scheme_name: str,
    bands: tuple[tuple[str, float, float], ...],
    sfreq: float,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    per_layer = {name: {"input": [], "output": []} for name in layer_names}
    band_names = tuple(name for name, _, _ in bands)
    capture = LayerCapture(model.eval(), layer_names)
    try:
        for epoch in x:
            components, _ = fft_band_components(epoch, sfreq=sfreq, bands=bands)
            input_cos = cosine_matrix(components)
            input_odi = offdiag_mean_abs(input_cos)
            outputs = capture(components.astype(np.float32), device=device)
            for name in layer_names:
                output_odi = offdiag_mean_abs(cosine_matrix(outputs[name]))
                per_layer[name]["input"].append(input_odi)
                per_layer[name]["output"].append(output_odi)
    finally:
        capture.close()
    out = {}
    for name, vals in per_layer.items():
        in_mean = float(np.mean(vals["input"]))
        out_mean = float(np.mean(vals["output"]))
        out[name] = {
            "band_scheme": scheme_name,
            "band_names": band_names,
            "n_epochs": int(len(x)),
            "input_mean_abs_offdiag_cosine_mean": in_mean,
            "output_mean_abs_offdiag_cosine_mean": out_mean,
            "output_mean_abs_offdiag_cosine_std": float(np.std(vals["output"])),
            "collapse_factor_mean": out_mean / max(in_mean, 1e-12),
        }
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    fractions = parse_floats(args.fractions)
    distance_metrics = parse_items(args.pair_metrics)
    scheme_names = parse_items(args.band_schemes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, _val_x, _val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
    sfreq = 1.0 / data["dt"]
    model, all_layer_names, model_source = arch_spec(
        args.arch,
        train_x.shape[1],
        train_x.shape[2],
        n_outputs=4,
        sfreq=sfreq,
    )
    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.arch, args.label_mode, args.seed)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    layer_names = compact_layers(args.arch, all_layer_names)
    batch_size = effective_batch_size(args.arch, args.batch_size)
    rng = np.random.default_rng(args.seed)
    pool_size = min(args.max_epochs, len(data["x_all"]))
    pool_idx = rng.choice(len(data["x_all"]), size=pool_size, replace=False)
    x_pool = data["x_all"][pool_idx].astype(np.float32)
    x_flat_pool = x_pool.reshape(pool_size, -1).astype(np.float32)
    schemes = band_schemes(scheme_names, train_x, sfreq)
    activations = capture_activations(model, x_pool, layer_names, device=device, batch_size=batch_size)

    pairwise_rows = []
    band_rows = []
    for fraction in fractions:
        n_subset = max(2, int(round(pool_size * fraction)))
        for repeat in range(args.repeats):
            rep_seed = args.seed * 100_000 + int(round(fraction * 1000)) * 100 + repeat
            rep_rng = np.random.default_rng(rep_seed)
            subset = rep_rng.choice(pool_size, size=n_subset, replace=False)
            x_flat = x_flat_pool[subset]
            x_subset = x_pool[subset]
            for layer in layer_names:
                z_flat = activations[layer][subset]
                for metric in distance_metrics:
                    row = pairwise_report(
                        x_flat,
                        z_flat,
                        metric=metric,
                        dt=data["dt"],
                        n_pairs=args.n_pairs,
                        seed=rep_seed,
                    )
                    row.update(
                        {
                            "arch": args.arch,
                            "seed": args.seed,
                            "label_mode": args.label_mode,
                            "layer": layer,
                            "fraction": fraction,
                            "repeat": repeat,
                            "n_subset": n_subset,
                        }
                    )
                    pairwise_rows.append(row)
            for scheme_name, bands in schemes.items():
                band_by_layer = band_odi_for_subset(
                    model,
                    x_subset,
                    layer_names,
                    scheme_name=scheme_name,
                    bands=bands,
                    sfreq=sfreq,
                    device=device,
                )
                for layer, row in band_by_layer.items():
                    row.update(
                        {
                            "arch": args.arch,
                            "seed": args.seed,
                            "label_mode": args.label_mode,
                            "layer": layer,
                            "fraction": fraction,
                            "repeat": repeat,
                            "n_subset": n_subset,
                        }
                    )
                    band_rows.append(row)
            print(f"done arch={args.arch} fraction={fraction:.2f} repeat={repeat}", flush=True)

    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "audit_pool_shape": list(x_pool.shape),
            "pool_indices": pool_idx.astype(int).tolist(),
            "dt": data["dt"],
            "sfreq": sfreq,
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "checkpoint": str(ckpt),
            "layer_names": layer_names,
        },
        "sensitivity": {
            "fractions": fractions,
            "repeats": args.repeats,
            "n_pairs": args.n_pairs,
            "distance_metrics": distance_metrics,
            "band_schemes": {
                name: [(band_name, low, high) for band_name, low, high in bands]
                for name, bands in schemes.items()
            },
        },
        "pairwise_rows": pairwise_rows,
        "band_rows": band_rows,
    }
    out = Path(args.out or f"reports/hsdd_sensitivity/bci2a_{args.arch}_{args.label_mode}_seed{args.seed}_sensitivity.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
