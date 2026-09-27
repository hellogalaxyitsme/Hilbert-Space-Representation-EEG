#!/usr/bin/env python3
"""Cross-dataset HSDD hyperparameter sensitivity on trained deep models."""

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
import torch
from torch import nn

from hsdd.io import write_json
from p300_utils import load_p300_cache
from run_bci2a_hsdd_sensitivity import (
    band_odi_for_subset,
    band_schemes,
    capture_activations,
    pairwise_report,
    parse_floats,
    parse_items,
)
from seediv_utils import load_seediv_cache
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import load_bci2a, make_train_val, set_seed
from train_sleepedf_arch_layer_audit import load_cache as load_sleepedf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bci2a", "sleepedf_full", "seediv", "p300"), required=True)
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-epochs", type=int, default=256)
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-pairs", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pair-metrics", default="euclidean,cosine,correlation,l1")
    parser.add_argument("--band-schemes", default="canonical,narrow_2hz,data_driven_equal_power")
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


def load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci2a":
        data = load_bci2a(Path(args.bci_root), tmin=0.0, tmax=3.0)
        train_x, train_y, _val_x, _val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
        return {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "x": data["x_all"].astype(np.float32),
            "train_x": train_x,
            "train_y": train_y,
            "sfreq": 1.0 / float(data["dt"]),
            "dt": float(data["dt"]),
            "n_outputs": 4,
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
            "train_x": data["train_x"],
            "train_y": train_y,
            "sfreq": float(data["sfreq"]),
            "dt": 1.0 / float(data["sfreq"]),
            "n_outputs": len(data["label_names"]),
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
            "train_x": data["train_x"],
            "train_y": train_y,
            "sfreq": float(data["sfreq"]),
            "dt": float(data["dt"]),
            "n_outputs": len(data["label_names"]),
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
        "train_x": data["train_x"],
        "train_y": train_y,
        "sfreq": float(data["sfreq"]),
        "dt": float(data["dt"]),
        "n_outputs": len(data["label_names"]),
    }


def checkpoint_candidates(dataset: str, arch: str, mode: str, seed: int) -> list[Path]:
    if dataset == "bci2a":
        if seed == 41:
            out = [Path(f"reports/{arch}_bci2a_{mode}.pt")]
            if mode == "true":
                out.append(Path(f"reports/{arch}_bci2a.pt"))
            return out
        return [Path(f"reports/{arch}_bci2a_{mode}_seed{seed}.pt")]
    if dataset in {"sleepedf_full", "seediv", "p300"}:
        return [Path(f"reports/{dataset}_{arch}_{mode}_seed{seed}.pt")]
    raise ValueError(dataset)


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    candidates = checkpoint_candidates(args.dataset, args.arch, args.label_mode, args.seed)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing checkpoint for {args.dataset}/{args.arch}/{args.label_mode}/seed{args.seed}: {candidates}")


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    fractions = parse_floats(args.fractions)
    distance_metrics = parse_items(args.pair_metrics)
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
    ckpt = checkpoint_path(args)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    requested_layer_names = compact_layers(args.arch, all_layer_names)
    layer_names = available_layer_names(model, requested_layer_names)
    dropped_layer_names = [name for name in requested_layer_names if name not in layer_names]
    if dropped_layer_names:
        print(f"dropped unavailable layers for sensitivity: {dropped_layer_names}", flush=True)
    batch_size = effective_batch_size(args.arch, args.batch_size)

    rng = np.random.default_rng(args.seed)
    pool_size = min(args.max_epochs, len(data["x"]))
    pool_idx = rng.choice(len(data["x"]), size=pool_size, replace=False)
    x_pool = data["x"][pool_idx].astype(np.float32)
    x_flat_pool = x_pool.reshape(pool_size, -1).astype(np.float32)
    schemes = band_schemes(scheme_names, data["train_x"], data["sfreq"])
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
                    row = pairwise_report(x_flat, z_flat, metric=metric, dt=data["dt"], n_pairs=args.n_pairs, seed=rep_seed)
                    row.update(
                        {
                            "dataset": args.dataset,
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
                    sfreq=data["sfreq"],
                    device=device,
                )
                for layer, row in band_by_layer.items():
                    row.update(
                        {
                            "dataset": args.dataset,
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
            print(f"done dataset={args.dataset} arch={args.arch} mode={args.label_mode} fraction={fraction:.2f} repeat={repeat}", flush=True)

    payload = {
        "dataset": {
            "name": args.dataset,
            "kind": data["kind"],
            "x_shape": list(data["x"].shape),
            "audit_pool_shape": list(x_pool.shape),
            "pool_indices": pool_idx.astype(int).tolist(),
            "dt": data["dt"],
            "sfreq": data["sfreq"],
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "checkpoint": str(ckpt),
            "requested_layer_names": requested_layer_names,
            "dropped_layer_names": dropped_layer_names,
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
    out = Path(args.out or f"reports/hsdd_sensitivity/{args.dataset}_{args.arch}_{args.label_mode}_seed{args.seed}_sensitivity.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
