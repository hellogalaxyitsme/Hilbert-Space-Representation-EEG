#!/usr/bin/env python3
"""Train one architecture on cached Sleep-EDF and run static HSDD audit."""

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
from train_bci2a_arch_layer_audit import (
    ATCNET_LAYER_NAMES,
    TSCEPTION_LAYER_NAMES,
    build_atcnet,
    build_tsception,
)
from train_eegconformer_bci2a_layer_audit import EEGCONFORMER_LAYER_NAMES, build_eegconformer
from train_eegnet_bci2a_layer_audit import (
    LAYER_NAMES as EEGNET_LAYER_NAMES,
    LayerCapture,
    evaluate,
    layer_band_orthogonality,
    layer_local_directional_distortion,
    layer_pairwise_isometry,
    set_seed,
    train_model,
)
from train_shallowconvnet_bci2a_layer_audit import SHALLOW_LAYER_NAMES, build_shallow
from train_eegnet_bci2a_layer_audit import build_eegnet


ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--cache", default="prepared/sleepedf_sc20_fpzcz_pzoz.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-recordings", type=int, default=4)
    parser.add_argument("--max-train-per-class", type=int, default=1600)
    parser.add_argument("--max-val-per-class", type=int, default=400)
    parser.add_argument("--n-pairs", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=90)
    parser.add_argument("--local-points", type=int, default=8)
    parser.add_argument("--local-directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def arch_spec(arch: str, n_chans: int, n_times: int, n_outputs: int):
    if arch == "eegnet":
        return build_eegnet(n_chans, n_times, n_outputs=n_outputs), EEGNET_LAYER_NAMES, "braindecode.models.EEGNet"
    if arch == "shallowconvnet":
        return build_shallow(n_chans, n_times, n_outputs=n_outputs), SHALLOW_LAYER_NAMES, "braindecode.models.ShallowFBCSPNet"
    if arch == "eegconformer":
        return build_eegconformer(n_chans, n_times, n_outputs=n_outputs), EEGCONFORMER_LAYER_NAMES, "braindecode.models.EEGConformer"
    if arch == "tsception":
        return build_tsception(n_chans, n_times, n_outputs=n_outputs, sfreq=100.0), TSCEPTION_LAYER_NAMES, "braindecode.models.TSception"
    if arch == "atcnet":
        return build_atcnet(n_chans, n_times, n_outputs=n_outputs, sfreq=100.0), ATCNET_LAYER_NAMES, "braindecode.models.ATCNet"
    raise ValueError(arch)


def balanced_indices(y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        if max_per_class > 0 and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def load_cache(path: Path, val_recordings: int, seed: int, max_train_per_class: int, max_val_per_class: int):
    data = np.load(path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    rid = data["recording_id"].astype(np.int64)
    sfreq = float(data["sfreq"][0])
    rec_names = [str(v) for v in data["recording_names"]]
    label_names = [str(v) for v in data["label_names"]]
    channel_names = [str(v) for v in data["channel_names"]]
    unique = sorted(set(rid.tolist()))
    val_ids = set(unique[-val_recordings:])
    val_mask = np.asarray([int(v) in val_ids for v in rid])
    train_mask = ~val_mask
    train_idx_all = np.flatnonzero(train_mask)
    val_idx_all = np.flatnonzero(val_mask)
    train_sub = balanced_indices(y[train_idx_all], max_train_per_class, seed)
    val_sub = balanced_indices(y[val_idx_all], max_val_per_class, seed)
    train_idx = train_idx_all[train_sub]
    val_idx = val_idx_all[val_sub]
    return {
        "x": x,
        "y": y,
        "recording_id": rid,
        "recording_names": rec_names,
        "label_names": label_names,
        "channel_names": channel_names,
        "sfreq": sfreq,
        "train_x": x[train_idx],
        "train_y": y[train_idx],
        "val_x": x[val_idx],
        "val_y": y[val_idx],
        "train_recordings": [rec_names[i] for i in unique if i not in val_ids],
        "val_recordings": [rec_names[i] for i in unique if i in val_ids],
    }


def label_counts(y: np.ndarray, label_names: list[str]) -> dict[str, int]:
    return {label_names[i]: int((y == i).sum()) for i in range(len(label_names))}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cache(
        Path(args.cache),
        args.val_recordings,
        args.seed,
        args.max_train_per_class,
        args.max_val_per_class,
    )
    if args.label_mode == "shuffled":
        rng = np.random.default_rng(args.seed)
        data["train_y"] = rng.permutation(data["train_y"])

    model, layer_names, model_source = arch_spec(
        args.arch,
        data["train_x"].shape[1],
        data["train_x"].shape[2],
        n_outputs=len(data["label_names"]),
    )
    model.to(device)
    print(model, flush=True)
    batch_size = args.batch_size
    if args.arch in {"eegconformer", "tsception", "atcnet"}:
        batch_size = min(batch_size, 32)

    if args.skip_training:
        val_loader = DataLoader(TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])), batch_size=batch_size)
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        history = [{"epoch": 0, "train_loss": None, "train_acc": None, "val_loss": val_loss, "val_acc": val_acc}]
        print(f"skip_training val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
    else:
        history = train_model(
            model,
            data["train_x"],
            data["train_y"],
            data["val_x"],
            data["val_y"],
            device=device,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    out = Path(args.out or f"reports/sleepedf_{args.arch}_layer_audit_{args.label_mode}.json")
    checkpoint = Path(args.checkpoint or f"reports/sleepedf_{args.arch}_{args.label_mode}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, checkpoint)

    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            data["x"],
            capture,
            layer_names=layer_names,
            device=device,
            dt=1.0 / data["sfreq"],
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
            dt=1.0 / data["sfreq"],
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
            "kind": "Sleep_EDF_sleep_cassette_cache",
            "cache": args.cache,
            "x_shape": list(data["x"].shape),
            "sfreq": data["sfreq"],
            "label_names": data["label_names"],
            "label_counts": label_counts(data["y"], data["label_names"]),
            "train_shape": list(data["train_x"].shape),
            "val_shape": list(data["val_x"].shape),
            "train_label_counts": label_counts(data["train_y"], data["label_names"]),
            "val_label_counts": label_counts(data["val_y"], data["label_names"]),
            "train_recordings": data["train_recordings"],
            "val_recordings": data["val_recordings"],
        },
        "model": {"arch": args.arch, "source": model_source, "layer_names": layer_names, "n_parameters": int(sum(p.numel() for p in model.parameters()))},
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
