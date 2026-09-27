#!/usr/bin/env python3
"""Per-layer band-power linear probes for Tier-2 redundancy checks."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from run_bci2a_hsdd_sensitivity import band_schemes, parse_items
from run_cka_rsa_dynamics import checkpoint_file, default_checkpoint_dir
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import LayerCapture, load_bci2a, make_train_val, set_seed
from train_sleepedf_arch_layer_audit import load_cache as load_sleepedf_cache
from seediv_utils import load_seediv_cache
from p300_utils import load_p300_cache


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
    parser.add_argument("--band-schemes", default="canonical")
    parser.add_argument("--layer-scope", choices=("compact", "all"), default="compact")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-per-class-probe", type=int, default=240)
    parser.add_argument("--max-val-per-class-probe", type=int, default=120)
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


def parse_checkpoints(spec: str | None, dataset: str, epochs: int) -> list[int]:
    if spec:
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
    return [epochs]


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def load_split(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci2a":
        data = load_bci2a(Path(args.bci_root), tmin=0.0, tmax=3.0)
        train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
        return {
            "kind": "BCI_IV_2a_preconverted_fif_cross_subject_A09T",
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "sfreq": 1.0 / data["dt"],
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
            "kind": "Sleep_EDF_full_sleep_cassette_recording_split",
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": data["sfreq"],
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
            "kind": "SEED_IV_raw_window_subject_split",
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": data["sfreq"],
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
        "kind": "BNCI2014_009_P300_raw_event_subject_split",
        "train_x": data["train_x"],
        "train_y": train_y,
        "val_x": data["val_x"],
        "val_y": data["val_y"],
        "sfreq": data["sfreq"],
        "n_outputs": len(data["label_names"]),
    }


def stratified_cap(x: np.ndarray, y: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if per_class <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        take = min(per_class, len(idx))
        selected.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return x[out], y[out]


def layer_bandpower_features(
    model: nn.Module,
    x: np.ndarray,
    layer_names: tuple[str, ...],
    bands: tuple[tuple[str, float, float], ...],
    sfreq: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    features = {layer: [] for layer in layer_names}
    capture = LayerCapture(model.eval(), layer_names)
    try:
        for epoch in x:
            components, _ = fft_band_components(epoch, sfreq=sfreq, bands=bands)
            outputs = capture(components.astype(np.float32), device=device)
            for layer in layer_names:
                z = outputs[layer].reshape(outputs[layer].shape[0], -1).astype(np.float64)
                power = np.log(np.mean(z * z, axis=1) + 1e-12)
                features[layer].append(power)
    finally:
        capture.close()
    return {layer: np.asarray(vals, dtype=np.float64) for layer, vals in features.items()}


def fit_probe(train_features: np.ndarray, train_y: np.ndarray, val_features: np.ndarray, val_y: np.ndarray) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto", C=1.0),
    )
    clf.fit(train_features, train_y)
    train_acc = float(clf.score(train_features, train_y))
    val_acc = float(clf.score(val_features, val_y))
    return {"bandpower_probe_train_acc": train_acc, "bandpower_probe_val_acc": val_acc}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dataset == "p300" and args.arch == "atcnet":
        raise ValueError("P300/ATCNet is excluded from reported-scope Tier-2 runs.")

    epochs = args.epochs if args.epochs is not None else default_epochs(args.dataset)
    checkpoints = parse_checkpoints(args.checkpoints, args.dataset, epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(args)
    train_x, train_y = stratified_cap(
        split["train_x"], split["train_y"], args.max_train_per_class_probe, args.seed
    )
    val_x, val_y = stratified_cap(
        split["val_x"], split["val_y"], args.max_val_per_class_probe, args.seed + 1
    )
    model, all_layer_names, model_source = arch_spec(
        args.arch,
        train_x.shape[1],
        train_x.shape[2],
        n_outputs=split["n_outputs"],
        sfreq=split["sfreq"],
    )
    requested_layers = available_layer_names(model, all_layer_names)
    if args.layer_scope == "compact":
        requested_layers = available_layer_names(model, compact_layers(args.arch, all_layer_names))
    model.to(device).eval()
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(
        "sleepedf_full" if args.dataset == "sleepedf_full" else args.dataset,
        args.arch,
        args.label_mode,
        args.seed,
    )
    schemes = band_schemes(parse_items(args.band_schemes), train_x, split["sfreq"])
    rows = []
    for epoch in checkpoints:
        ckpt = checkpoint_file(ckpt_dir, args.arch, args.label_mode, epoch)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        for scheme_name, bands in schemes.items():
            train_features = layer_bandpower_features(model, train_x, requested_layers, bands, split["sfreq"], device)
            val_features = layer_bandpower_features(model, val_x, requested_layers, bands, split["sfreq"], device)
            for layer in requested_layers:
                metrics = fit_probe(train_features[layer], train_y, val_features[layer], val_y)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "arch": args.arch,
                        "mode": args.label_mode,
                        "seed": args.seed,
                        "epoch": epoch,
                        "checkpoint": str(ckpt),
                        "band_scheme": scheme_name,
                        "band_names": tuple(name for name, _, _ in bands),
                        "n_bands": len(bands),
                        "layer": layer,
                        "n_train": len(train_x),
                        "n_val": len(val_x),
                        **metrics,
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
            "kind": split["kind"],
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "sfreq": split["sfreq"],
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
            "probe": "multinomial logistic regression on per-band log mean-square layer responses",
        },
        "rows": rows,
    }
    out = Path(
        args.out
        or f"reports/tier2_bandpower_probes/{args.dataset}_{args.arch}_{args.label_mode}_seed{args.seed}_bandpower_probe.json"
    )
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
