#!/usr/bin/env python3
"""Compute CKA/RSA at the same saved checkpoints used by HSDD dynamics."""

from __future__ import annotations

import argparse
import json
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
from torch import nn

from hsdd.diagnostics import _spearman
from hsdd.io import write_json
from p300_utils import load_p300_cache
from seediv_utils import load_seediv_cache
from train_bci2a_arch_layer_audit import ARCHES, arch_spec as bci_arch_spec, effective_batch_size
from train_eegnet_bci2a_dynamics import parse_checkpoints
from train_eegnet_bci2a_layer_audit import LayerCapture, load_bci2a, make_train_val, set_seed
from train_sleepedf_arch_layer_audit import arch_spec as sleepedf_arch_spec, load_cache as load_sleepedf_cache


DATASETS = ("bci2a", "sleepedf", "seediv", "p300")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--dynamics-report", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--n-samples", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-source", choices=("val", "audit"), default="val")
    parser.add_argument("--root", default=None)
    parser.add_argument("--sleepedf-cache", default="prepared/sleepedf_sc20_fpzcz_pzoz.npz")
    parser.add_argument("--seediv-cache", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--p300-cache", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--sleepedf-val-recordings", type=int, default=4)
    parser.add_argument("--seediv-val-subjects", default="13,14,15")
    parser.add_argument("--p300-val-subjects", default="9,10")
    parser.add_argument("--sleepedf-max-train-per-class", type=int, default=1600)
    parser.add_argument("--sleepedf-max-val-per-class", type=int, default=400)
    parser.add_argument("--seediv-max-train-per-class", type=int, default=800)
    parser.add_argument("--seediv-max-val-per-class", type=int, default=200)
    parser.add_argument("--p300-max-train-per-class", type=int, default=288)
    parser.add_argument("--p300-max-val-per-class", type=int, default=72)
    return parser.parse_args()


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def default_epochs(dataset: str) -> int:
    return 30 if dataset == "bci2a" else 20


def default_checkpoints(dataset: str) -> str:
    return "0,1,2,5,10,20,30" if dataset == "bci2a" else "0,1,2,5,10,20"


def no_suffix_seed41_path(path: Path, seed: int) -> Path:
    if seed == 41:
        alt = Path(str(path).replace("_seed41", ""))
        if alt.exists():
            return alt
    return path


def default_dynamics_report(dataset: str, arch: str, mode: str, seed: int) -> Path:
    if dataset == "bci2a":
        path = Path(f"reports/{arch}_bci2a_dynamics_{mode}_seed{seed}.json")
    else:
        path = Path(f"reports/{dataset}_{arch}_dynamics_{mode}_seed{seed}.json")
    return no_suffix_seed41_path(path, seed)


def default_checkpoint_dir(dataset: str, arch: str, mode: str, seed: int) -> Path:
    if dataset == "bci2a":
        path = Path(f"reports/{arch}_bci2a_dynamics_{mode}_seed{seed}_checkpoints")
        if seed == 41 and arch in {"eegnet", "shallowconvnet"}:
            legacy = Path(f"reports/{arch}_dynamics_{mode}")
            if legacy.exists():
                return legacy
    else:
        path = Path(f"reports/{dataset}_{arch}_dynamics_{mode}_seed{seed}_checkpoints")
    return no_suffix_seed41_path(path, seed)


def checkpoint_file(checkpoint_dir: Path, arch: str, mode: str, epoch: int) -> Path:
    return checkpoint_dir / f"{arch}_{mode}_epoch_{epoch:03d}.pt"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci2a":
        data = load_bci2a(Path(args.root), tmin=0.0, tmax=3.0)
        train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
        model, layer_names, model_source = bci_arch_spec(
            args.arch,
            train_x.shape[1],
            train_x.shape[2],
            n_outputs=4,
            sfreq=1.0 / data["dt"],
        )
        return {
            "x_audit": data["x_all"],
            "y_audit": data["labels_all"],
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "sfreq": 1.0 / data["dt"],
            "dt": data["dt"],
            "n_outputs": 4,
            "model": model,
            "layer_names": layer_names,
            "model_source": model_source,
        }

    if args.dataset == "sleepedf":
        data = load_sleepedf_cache(
            Path(args.sleepedf_cache),
            args.sleepedf_val_recordings,
            args.seed,
            args.sleepedf_max_train_per_class,
            args.sleepedf_max_val_per_class,
        )
        train_y = data["train_y"].copy()
        if args.label_mode == "shuffled":
            train_y = np.random.default_rng(args.seed).permutation(train_y)
        model, layer_names, model_source = sleepedf_arch_spec(
            args.arch,
            data["train_x"].shape[1],
            data["train_x"].shape[2],
            n_outputs=len(data["label_names"]),
        )
        return {
            "x_audit": data["x"],
            "y_audit": data["y"],
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": data["sfreq"],
            "dt": 1.0 / data["sfreq"],
            "n_outputs": len(data["label_names"]),
            "model": model,
            "layer_names": layer_names,
            "model_source": model_source,
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
        model, layer_names, model_source = bci_arch_spec(
            args.arch,
            data["train_x"].shape[1],
            data["train_x"].shape[2],
            n_outputs=len(data["label_names"]),
            sfreq=data["sfreq"],
        )
        return {
            "x_audit": data["x"],
            "y_audit": data["y"],
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": data["sfreq"],
            "dt": data["dt"],
            "n_outputs": len(data["label_names"]),
            "model": model,
            "layer_names": layer_names,
            "model_source": model_source,
        }

    if args.dataset == "p300":
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
        model, layer_names, model_source = bci_arch_spec(
            args.arch,
            data["train_x"].shape[1],
            data["train_x"].shape[2],
            n_outputs=len(data["label_names"]),
            sfreq=data["sfreq"],
        )
        layer_names = available_layer_names(model, layer_names)
        return {
            "x_audit": data["x"],
            "y_audit": data["y"],
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "sfreq": data["sfreq"],
            "dt": data["dt"],
            "n_outputs": len(data["label_names"]),
            "model": model,
            "layer_names": layer_names,
            "model_source": model_source,
        }

    raise ValueError(args.dataset)


def stratified_sample(x: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if n_samples <= 0 or n_samples >= len(x):
        idx = np.arange(len(x))
    else:
        selected = []
        labels = sorted(set(y.tolist()))
        per = max(1, n_samples // max(len(labels), 1))
        for label in labels:
            label_idx = np.flatnonzero(y == label)
            if len(label_idx) == 0:
                continue
            take = min(per, len(label_idx))
            selected.append(rng.choice(label_idx, size=take, replace=False))
        idx = np.concatenate(selected) if selected else rng.choice(len(x), size=n_samples, replace=False)
        if len(idx) < n_samples:
            remaining = np.setdiff1d(np.arange(len(x)), idx, assume_unique=False)
            if len(remaining) > 0:
                extra = rng.choice(remaining, size=min(n_samples - len(idx), len(remaining)), replace=False)
                idx = np.concatenate([idx, extra])
        rng.shuffle(idx)
    return x[idx].astype(np.float32), y[idx].astype(np.int64), idx.astype(int).tolist()


def capture_activations(
    model: nn.Module,
    x: np.ndarray,
    layer_names: tuple[str, ...],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    capture = LayerCapture(model.eval(), layer_names)
    chunks = {name: [] for name in layer_names}
    try:
        for start in range(0, len(x), batch_size):
            outputs = capture(x[start : start + batch_size], device=device)
            for name in layer_names:
                chunks[name].append(outputs[name].reshape(outputs[name].shape[0], -1).astype(np.float64))
    finally:
        capture.close()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def centered_gram(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    gram -= gram.mean(axis=0, keepdims=True)
    gram -= gram.mean(axis=1, keepdims=True)
    gram += gram.mean()
    return gram


def linear_cka(x: np.ndarray, z: np.ndarray) -> float:
    k = centered_gram(x)
    l = centered_gram(z)
    num = float(np.sum(k * l))
    den = math.sqrt(max(float(np.sum(k * k)), 0.0) * max(float(np.sum(l * l)), 0.0))
    if den <= 1e-12:
        return float("nan")
    return num / den


def euclidean_rdm_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    gram = x @ x.T
    sq = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0)
    dist = np.sqrt(sq)
    tri = np.triu_indices(dist.shape[0], k=1)
    return dist[tri]


def label_rdm_vector(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    same = y[:, None] == y[None, :]
    dist = (~same).astype(np.float64)
    tri = np.triu_indices(dist.shape[0], k=1)
    return dist[tri]


def rank_estimate(x: np.ndarray, tol: float = 1e-6) -> int:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    eig = np.linalg.eigvalsh(gram)
    if len(eig) == 0:
        return 0
    eig = np.maximum(eig, 0.0)
    top = max(float(np.max(eig)), 1e-12)
    return int(np.sum(eig > tol * top))


def metrics_for_layers(x_flat: np.ndarray, y: np.ndarray, activations: dict[str, np.ndarray]) -> dict[str, dict[str, float | int | list[int]]]:
    input_rdm = euclidean_rdm_vector(x_flat)
    label_rdm = label_rdm_vector(y)
    out: dict[str, dict[str, float | int | list[int]]] = {}
    for name, z in activations.items():
        z_rdm = euclidean_rdm_vector(z)
        out[name] = {
            "linear_cka_input": linear_cka(x_flat, z),
            "rsa_input_spearman": _spearman(input_rdm, z_rdm),
            "rsa_label_spearman": _spearman(label_rdm, z_rdm),
            "activation_shape": [int(v) for v in z.shape],
            "activation_rank": rank_estimate(z),
        }
    return out


def audit_lookup(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["epoch"]): row for row in report.get("audits", [])}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    epochs = args.epochs if args.epochs is not None else default_epochs(args.dataset)
    checkpoints_raw = args.checkpoints or default_checkpoints(args.dataset)
    checkpoints = parse_checkpoints(checkpoints_raw, epochs)

    dynamics_report = Path(args.dynamics_report) if args.dynamics_report else default_dynamics_report(args.dataset, args.arch, args.label_mode, args.seed)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(args.dataset, args.arch, args.label_mode, args.seed)
    out = Path(args.out or f"reports/cka_rsa/{args.dataset}_{args.arch}_dynamics_{args.label_mode}_seed{args.seed}_cka_rsa.json")

    if not dynamics_report.exists():
        raise FileNotFoundError(f"missing dynamics report: {dynamics_report}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"missing checkpoint dir: {checkpoint_dir}")

    dynamics = load_json(dynamics_report)
    dynamics_by_epoch = audit_lookup(dynamics)
    data = load_dataset(args)
    model: nn.Module = data["model"]
    layer_names = available_layer_names(model, tuple(data["layer_names"]))

    source_x = data["val_x"] if args.sample_source == "val" else data["x_audit"]
    source_y = data["val_y"] if args.sample_source == "val" else data["y_audit"]
    sample_x, sample_y, sample_indices = stratified_sample(source_x, source_y, args.n_samples, args.seed)
    input_flat = sample_x.reshape(sample_x.shape[0], -1).astype(np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)

    reports = []
    for epoch in checkpoints:
        ckpt = checkpoint_file(checkpoint_dir, args.arch, args.label_mode, epoch)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        activations = capture_activations(model, sample_x, layer_names, device=device, batch_size=batch_size)
        layer_metrics = metrics_for_layers(input_flat, sample_y, activations)
        dyn = dynamics_by_epoch.get(epoch, {})
        reports.append(
            {
                "epoch": int(epoch),
                "checkpoint": str(ckpt),
                "train_acc": None if dyn.get("train_acc") is None else float(dyn.get("train_acc")),
                "val_acc": None if dyn.get("val_acc") is None else float(dyn.get("val_acc")),
                "layers": layer_metrics,
            }
        )
        print(
            f"done dataset={args.dataset} arch={args.arch} labels={args.label_mode} "
            f"seed={args.seed} epoch={epoch:03d}",
            flush=True,
        )

    payload = {
        "dataset": {
            "name": args.dataset,
            "sample_source": args.sample_source,
            "n_samples": int(len(sample_x)),
            "sample_indices": sample_indices,
            "label_counts": {str(label): int((sample_y == label).sum()) for label in sorted(set(sample_y.tolist()))},
            "x_shape": list(sample_x.shape),
        },
        "model": {
            "arch": args.arch,
            "source": data["model_source"],
            "layer_names": layer_names,
        },
        "run": {
            "label_mode": args.label_mode,
            "seed": args.seed,
            "epochs": epochs,
            "checkpoints": checkpoints,
            "dynamics_report": str(dynamics_report),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": {
                "linear_cka_input": "linear CKA between raw input epochs and layer activations on the same sampled epochs",
                "rsa_input_spearman": "Spearman correlation between raw-input Euclidean RDM and layer Euclidean RDM",
                "rsa_label_spearman": "Spearman correlation between label-different RDM and layer Euclidean RDM",
            },
        },
        "audits": reports,
    }
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
