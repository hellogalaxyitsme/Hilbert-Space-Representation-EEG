#!/usr/bin/env python3
"""Train one architecture on SEED-IV raw EEG windows and run static HSDD audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsdd.io import write_json
from seediv_utils import label_counts, load_seediv_cache
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import (
    LayerCapture,
    evaluate,
    layer_band_orthogonality,
    layer_local_directional_distortion,
    layer_pairwise_isometry,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--cache", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-subjects", default="13,14,15")
    parser.add_argument("--max-train-per-class", type=int, default=800)
    parser.add_argument("--max-val-per-class", type=int, default=200)
    parser.add_argument("--n-pairs", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=90)
    parser.add_argument("--local-points", type=int, default=8)
    parser.add_argument("--local-directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_seediv_cache(
        Path(args.cache),
        val_subjects=args.val_subjects,
        seed=args.seed,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
    )
    train_y = data["train_y"].copy()
    if args.label_mode == "shuffled":
        train_y = np.random.default_rng(args.seed).permutation(train_y)

    model, layer_names, model_source = arch_spec(
        args.arch,
        data["train_x"].shape[1],
        data["train_x"].shape[2],
        n_outputs=len(data["label_names"]),
        sfreq=data["sfreq"],
    )
    print(model, flush=True)
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)

    if args.skip_training:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        history = [{"epoch": 0, "train_loss": None, "train_acc": None, "val_loss": val_loss, "val_acc": val_acc}]
        print(f"skip_training val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
    else:
        history = train_model(
            model,
            data["train_x"],
            train_y,
            data["val_x"],
            data["val_y"],
            device=device,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    out = Path(args.out or f"reports/seediv_{args.arch}_layer_audit_{args.label_mode}.json")
    checkpoint = Path(args.checkpoint or f"reports/seediv_{args.arch}_{args.label_mode}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, checkpoint)

    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            data["x"],
            capture,
            layer_names=layer_names,
            device=device,
            dt=data["dt"],
            n_pairs=args.n_pairs,
            pair_batch_size=min(args.pair_batch_size, batch_size),
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            data["x"],
            capture,
            layer_names=layer_names,
            device=device,
            sfreq=data["sfreq"],
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            data["x"],
            capture,
            layer_names=layer_names,
            device=device,
            dt=data["dt"],
            n_points=args.local_points,
            n_directions=args.local_directions,
            eps=1e-3,
            seed=args.seed,
        )
    finally:
        capture.close()

    layers = {
        name: {
            "pairwise_isometry": pairwise[name],
            "band_orthogonality": band[name],
            "local_directional_distortion": local[name],
        }
        for name in layer_names
    }
    payload = {
        "dataset": {
            "kind": "SEED_IV_raw_eeg_window_cache",
            "cache": args.cache,
            "x_shape": list(data["x"].shape),
            "sfreq": data["sfreq"],
            "channel_names": data["channel_names"],
            "label_names": data["label_names"],
            "label_counts": label_counts(data["y"], data["label_names"]),
            "train_shape": list(data["train_x"].shape),
            "val_shape": list(data["val_x"].shape),
            "train_label_counts": label_counts(train_y, data["label_names"]),
            "val_label_counts": label_counts(data["val_y"], data["label_names"]),
            "train_subjects": data["train_subjects"],
            "val_subjects": data["val_subjects"],
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "checkpoint": str(checkpoint),
            "layer_names": layer_names,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "history": history,
            "best_val_acc": float(max(row["val_acc"] for row in history)),
            "final_val_acc": float(history[-1]["val_acc"]),
            "final_train_acc": None if history[-1]["train_acc"] is None else float(history[-1]["train_acc"]),
            "label_mode": args.label_mode,
            "skip_training": bool(args.skip_training),
        },
        "layers": layers,
    }
    write_json(out, payload)
    print("wrote", out, flush=True)
    print("final_train_acc", payload["training"]["final_train_acc"], "best_val_acc", f"{payload['training']['best_val_acc']:.3f}", flush=True)
    for name in layer_names:
        print(name, "spearman=", f"{layers[name]['pairwise_isometry']['distance_spearman']:.4f}", "odi=", f"{layers[name]['band_orthogonality']['output_mean_abs_offdiag_cosine_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
