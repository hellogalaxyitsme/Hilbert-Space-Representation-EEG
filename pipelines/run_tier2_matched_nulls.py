#!/usr/bin/env python3
"""Dimension-matched HSDD nulls for audited deep EEG layers.

This is a no-new-training Tier-2 analysis. It loads existing dynamics
checkpoints, recomputes layer ODI for band-isolated inputs, and compares each
layer to output-dimension-matched null maps.
"""

from __future__ import annotations

import argparse
import math
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
from scipy.special import gammaln
from torch import nn

from hsdd.diagnostics import cosine_matrix, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from run_bci2a_hsdd_sensitivity import band_schemes, parse_items
from run_cka_rsa_dynamics import checkpoint_file, default_checkpoint_dir
from run_cross_dataset_hsdd_sensitivity import load_dataset
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegconformer_bci2a_layer_audit import EEGCONFORMER_LAYER_NAMES
from train_eegnet_bci2a_layer_audit import LayerCapture, set_seed


DATASETS = ("bci2a", "sleepedf_full", "seediv", "p300")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--band-schemes", default="canonical,narrow_2hz,data_driven_equal_power")
    parser.add_argument("--layer-scope", choices=("compact", "all"), default="compact")
    parser.add_argument("--null-repeats", type=int, default=8)
    parser.add_argument("--projection-simulation-max-dim", type=int, default=512)
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


def default_epochs(dataset: str) -> int:
    return 30 if dataset == "bci2a" else 20


def default_checkpoints(dataset: str, epochs: int) -> list[int]:
    base = [0, 1, 2, 5, 10, 20]
    if dataset == "bci2a":
        base.append(30)
    return [epoch for epoch in base if epoch <= epochs]


def parse_checkpoints(spec: str | None, dataset: str, epochs: int) -> list[int]:
    if not spec:
        return default_checkpoints(dataset, epochs)
    out = []
    for item in spec.replace(",", " ").split():
        token = item.strip().lower()
        if token in {"final", "last"}:
            value = epochs
        else:
            value = int(token)
        if value <= epochs:
            out.append(value)
    return sorted(set(out))


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def random_vector_expected_abs_cos(dim: int) -> float:
    if dim <= 1:
        return 1.0
    # E|cos(theta)| for two independent random unit vectors in R^dim.
    return float(math.exp(gammaln(dim / 2.0) - 0.5 * math.log(math.pi) - gammaln((dim + 1) / 2.0)))


def odi_from_features(features: np.ndarray) -> float:
    return offdiag_mean_abs(cosine_matrix(features))


def gaussian_projection_odi(
    x_flat: np.ndarray,
    out_dim: int,
    rng: np.random.Generator,
    *,
    simulation_max_dim: int,
) -> float:
    if out_dim >= simulation_max_dim:
        return odi_from_features(x_flat)
    gram = x_flat @ x_flat.T
    cov = gram / max(float(out_dim), 1.0)
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(cov.shape[0]) * 1e-8
    y = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=max(out_dim, 1)).T
    return odi_from_features(y)


def orthogonal_projection_odi(
    x_flat: np.ndarray,
    out_dim: int,
    rng: np.random.Generator,
    *,
    simulation_max_dim: int,
) -> float:
    in_dim = x_flat.shape[1]
    if out_dim >= in_dim:
        return odi_from_features(x_flat)
    if out_dim >= simulation_max_dim:
        return odi_from_features(x_flat)
    # A signed semi-orthogonal sketch is the fast finite-dimensional surrogate
    # for a random projection when out_dim is small. QR on the full EEG input
    # dimension is prohibitively slow inside the layer/checkpoint loop.
    sketch = rng.choice((-1.0, 1.0), size=(in_dim, max(out_dim, 1))) / math.sqrt(max(out_dim, 1))
    return odi_from_features(x_flat @ sketch)


def pca_projection_odi(x_flat: np.ndarray, out_dim: int, pca_basis: np.ndarray | None) -> float:
    if pca_basis is None or out_dim <= 0:
        return float("nan")
    n_components = min(out_dim, pca_basis.shape[1])
    if n_components <= 0:
        return float("nan")
    return odi_from_features(x_flat @ pca_basis[:, :n_components])


def pca_basis(x_pool: np.ndarray) -> np.ndarray | None:
    x = x_pool.reshape(x_pool.shape[0], -1).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if min(x.shape) < 2:
        return None
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return vt.T


def capture_band_outputs(
    model: nn.Module,
    components: np.ndarray,
    layer_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, np.ndarray]:
    capture = LayerCapture(model.eval(), layer_names)
    try:
        outputs = capture(components.astype(np.float32), device=device)
    finally:
        capture.close()
    return {name: outputs[name].reshape(outputs[name].shape[0], -1).astype(np.float64) for name in layer_names}


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dataset == "p300" and args.arch == "atcnet":
        raise ValueError("P300/ATCNet is excluded from reported-scope Tier-2 runs.")

    epochs = args.epochs if args.epochs is not None else default_epochs(args.dataset)
    checkpoints = parse_checkpoints(args.checkpoints, args.dataset, epochs)
    scheme_names = parse_items(args.band_schemes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args)
    model, all_layer_names, model_source = arch_spec(
        args.arch,
        data["train_x"].shape[1],
        data["train_x"].shape[2],
        n_outputs=data["n_outputs"],
        sfreq=data["sfreq"],
    )
    untrained_model, _, _ = arch_spec(
        args.arch,
        data["train_x"].shape[1],
        data["train_x"].shape[2],
        n_outputs=data["n_outputs"],
        sfreq=data["sfreq"],
    )
    requested_layers = available_layer_names(model, all_layer_names)
    if args.layer_scope == "compact":
        requested_layers = available_layer_names(model, compact_layers(args.arch, all_layer_names))
    model.to(device).eval()
    untrained_model.to(device).eval()
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(
        "sleepedf_full" if args.dataset == "sleepedf_full" else args.dataset,
        args.arch,
        args.label_mode,
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    pool_size = min(args.n_epochs, len(data["x"]))
    pool_idx = rng.choice(len(data["x"]), size=pool_size, replace=False)
    x_pool = data["x"][pool_idx].astype(np.float32)
    basis = pca_basis(x_pool)
    schemes = band_schemes(scheme_names, data["train_x"], data["sfreq"])

    audits: list[dict[str, Any]] = []
    for epoch in checkpoints:
        ckpt = checkpoint_file(ckpt_dir, args.arch, args.label_mode, epoch)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        for scheme_name, bands in schemes.items():
            band_names = tuple(name for name, _, _ in bands)
            components_by_epoch = []
            input_odi_values = []
            x_flat_by_epoch = []
            for eeg_epoch in x_pool:
                components, _ = fft_band_components(eeg_epoch, sfreq=data["sfreq"], bands=bands)
                components = components.astype(np.float32, copy=False)
                x_flat = components.reshape(components.shape[0], -1).astype(np.float64)
                components_by_epoch.append(components)
                x_flat_by_epoch.append(x_flat)
                input_odi_values.append(odi_from_features(x_flat))
            all_components = np.concatenate(components_by_epoch, axis=0)
            trained_all = capture_band_outputs(model, all_components, requested_layers, device)
            untrained_all = capture_band_outputs(untrained_model, all_components, requested_layers, device)
            layer_actual = {name: [] for name in requested_layers}
            layer_untrained = {name: [] for name in requested_layers}
            null_values: dict[str, dict[str, list[float]]] = {
                name: {"gaussian": [], "orthogonal": [], "pca": []} for name in requested_layers
            }
            layer_dims: dict[str, int] = {}
            n_bands = len(band_names)
            for local_i, x_flat in enumerate(x_flat_by_epoch):
                for layer in requested_layers:
                    trained_z = trained_all[layer][local_i * n_bands : (local_i + 1) * n_bands]
                    untrained_z = untrained_all[layer][local_i * n_bands : (local_i + 1) * n_bands]
                    actual = odi_from_features(trained_z)
                    untrained = odi_from_features(untrained_z)
                    out_dim = int(trained_z.shape[1])
                    layer_dims[layer] = out_dim
                    layer_actual[layer].append(actual)
                    layer_untrained[layer].append(untrained)
                    for repeat in range(args.null_repeats):
                        null_seed = args.seed * 1_000_000 + epoch * 10_000 + local_i * 100 + repeat
                        null_rng = np.random.default_rng(null_seed)
                        null_values[layer]["gaussian"].append(
                            gaussian_projection_odi(
                                x_flat,
                                out_dim,
                                null_rng,
                                simulation_max_dim=args.projection_simulation_max_dim,
                            )
                        )
                        null_values[layer]["orthogonal"].append(
                            orthogonal_projection_odi(
                                x_flat,
                                out_dim,
                                null_rng,
                                simulation_max_dim=args.projection_simulation_max_dim,
                            )
                        )
                        null_values[layer]["pca"].append(pca_projection_odi(x_flat, out_dim, basis))
            input_odi_mean, input_odi_std = summarize(input_odi_values)
            for layer in requested_layers:
                actual_mean, actual_std = summarize(layer_actual[layer])
                untrained_mean, untrained_std = summarize(layer_untrained[layer])
                null_means = {name: summarize(vals)[0] for name, vals in null_values[layer].items()}
                null_stds = {name: summarize(vals)[1] for name, vals in null_values[layer].items()}
                matched_null_mean = float(np.nanmean([null_means["gaussian"], null_means["orthogonal"], null_means["pca"], untrained_mean]))
                expected_chance = random_vector_expected_abs_cos(layer_dims[layer])
                audits.append(
                    {
                        "dataset": args.dataset,
                        "arch": args.arch,
                        "mode": args.label_mode,
                        "seed": args.seed,
                        "epoch": epoch,
                        "checkpoint": str(ckpt),
                        "band_scheme": scheme_name,
                        "band_names": band_names,
                        "n_bands": len(band_names),
                        "n_epochs": pool_size,
                        "layer": layer,
                        "output_dim": layer_dims[layer],
                        "input_odi_mean": input_odi_mean,
                        "input_odi_std": input_odi_std,
                        "actual_odi_mean": actual_mean,
                        "actual_odi_std": actual_std,
                        "gaussian_null_odi_mean": null_means["gaussian"],
                        "gaussian_null_odi_std": null_stds["gaussian"],
                        "orthogonal_null_odi_mean": null_means["orthogonal"],
                        "orthogonal_null_odi_std": null_stds["orthogonal"],
                        "pca_null_odi_mean": null_means["pca"],
                        "pca_null_odi_std": null_stds["pca"],
                        "untrained_same_arch_odi_mean": untrained_mean,
                        "untrained_same_arch_odi_std": untrained_std,
                        "matched_null_odi_mean": matched_null_mean,
                        "delta_odi_vs_matched_null": actual_mean - matched_null_mean,
                        "delta_odi_vs_untrained": actual_mean - untrained_mean,
                        "expected_odi_random_vectors": expected_chance,
                        "odi_over_expected_random": actual_mean / max(expected_chance, 1e-12),
                    }
                )
            print(
                f"done dataset={args.dataset} arch={args.arch} mode={args.label_mode} "
                f"seed={args.seed} epoch={epoch} scheme={scheme_name}",
                flush=True,
            )

    payload = {
        "dataset": {
            "name": args.dataset,
            "kind": data["kind"],
            "x_shape": list(data["x"].shape),
            "audit_pool_shape": list(x_pool.shape),
            "pool_indices": pool_idx.astype(int).tolist(),
            "sfreq": data["sfreq"],
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "layer_scope": args.layer_scope,
            "layer_names": requested_layers,
        },
        "run": {
            "label_mode": args.label_mode,
            "seed": args.seed,
            "epochs": epochs,
            "checkpoints": checkpoints,
            "band_schemes": {
                name: [(band_name, low, high) for band_name, low, high in bands]
                for name, bands in schemes.items()
            },
            "null_repeats": args.null_repeats,
            "projection_simulation_max_dim": args.projection_simulation_max_dim,
            "large_projection_note": "For matched output dimensions at or above projection_simulation_max_dim, Gaussian/orthogonal ODI uses the high-dimensional concentration limit, equal to input band ODI.",
            "nulls": [
                "random Gaussian projection at matched output dimension",
                "random orthogonal/semi-orthogonal projection at matched output dimension",
                "PCA projection from the same audit pool, truncated to matched output dimension",
                "untrained same architecture at the same layer",
            ],
        },
        "rows": audits,
    }
    out = Path(
        args.out
        or f"reports/tier2_matched_nulls/{args.dataset}_{args.arch}_{args.label_mode}_seed{args.seed}_matched_nulls.json"
    )
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
