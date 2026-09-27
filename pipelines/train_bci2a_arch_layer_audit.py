#!/usr/bin/env python3
"""Train one Braindecode architecture on BCI IV 2a and run static HSDD audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsdd.io import write_json
from train_eegnet_bci2a_layer_audit import (
    LAYER_NAMES as EEGNET_LAYER_NAMES,
    LayerCapture,
    build_eegnet,
    evaluate,
    label_counts,
    layer_band_orthogonality,
    layer_local_directional_distortion,
    layer_pairwise_isometry,
    load_bci2a,
    make_train_val,
    patch_braindecode_moabb_import,
    set_seed,
    train_model,
)
from train_eegconformer_bci2a_layer_audit import (
    EEGCONFORMER_LAYER_NAMES,
    build_eegconformer,
)
from train_shallowconvnet_bci2a_layer_audit import SHALLOW_LAYER_NAMES, build_shallow


TSCEPTION_LAYER_NAMES = (
    "temporal_blocks.0",
    "temporal_blocks.1",
    "temporal_blocks.2",
    "batch_temporal_lay",
    "spatial_block_1",
    "spatial_block_2",
    "batch_spatial_lay",
    "dense_layer",
    "final_layer",
    "logits",
)

ATCNET_LAYER_NAMES = (
    "conv_block",
    "attention_blocks.0",
    "attention_blocks.2",
    "attention_blocks.4",
    "temporal_conv_nets.0",
    "temporal_conv_nets.2",
    "temporal_conv_nets.4",
    "final_layer.0",
    "final_layer.4",
    "logits",
)

ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")


def build_tsception(
    n_chans: int,
    n_times: int,
    n_outputs: int = 4,
    sfreq: float = 250.0,
) -> nn.Module:
    patch_braindecode_moabb_import()
    from braindecode.models import TSception

    return TSception(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )


def build_atcnet(
    n_chans: int,
    n_times: int,
    n_outputs: int = 4,
    sfreq: float = 250.0,
) -> nn.Module:
    patch_braindecode_moabb_import()
    from braindecode.models import ATCNet

    return ATCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )


def arch_spec(
    arch: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
) -> tuple[nn.Module, tuple[str, ...], str]:
    if arch == "eegnet":
        return (
            build_eegnet(n_chans, n_times, n_outputs=n_outputs),
            EEGNET_LAYER_NAMES,
            "braindecode.models.EEGNet",
        )
    if arch == "shallowconvnet":
        return (
            build_shallow(n_chans, n_times, n_outputs=n_outputs),
            SHALLOW_LAYER_NAMES,
            "braindecode.models.ShallowFBCSPNet",
        )
    if arch == "eegconformer":
        return (
            build_eegconformer(n_chans, n_times, n_outputs=n_outputs),
            EEGCONFORMER_LAYER_NAMES,
            "braindecode.models.EEGConformer",
        )
    if arch == "tsception":
        return (
            build_tsception(n_chans, n_times, n_outputs=n_outputs, sfreq=sfreq),
            TSCEPTION_LAYER_NAMES,
            "braindecode.models.TSception",
        )
    if arch == "atcnet":
        return (
            build_atcnet(n_chans, n_times, n_outputs=n_outputs, sfreq=sfreq),
            ATCNET_LAYER_NAMES,
            "braindecode.models.ATCNet",
        )
    raise ValueError(arch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-pairs", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=180)
    parser.add_argument("--local-points", type=int, default=12)
    parser.add_argument("--local-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def effective_batch_size(arch: str, requested: int) -> int:
    if arch in {"eegconformer", "tsception", "atcnet"}:
        return min(requested, 32)
    return requested


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, val_x, val_y = make_train_val(
        data,
        label_mode=args.label_mode,
        seed=args.seed,
    )
    sfreq = 1.0 / data["dt"]
    model, layer_names, model_source = arch_spec(
        args.arch,
        train_x.shape[1],
        train_x.shape[2],
        n_outputs=4,
        sfreq=sfreq,
    )
    print(model, flush=True)
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)

    if args.skip_training:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        history = [
            {
                "epoch": 0,
                "train_loss": None,
                "train_acc": None,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        ]
        print(f"skip_training val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
    else:
        history = train_model(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            device=device,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    out = Path(args.out or f"reports/{args.arch}_bci2a_layer_audit_{args.label_mode}.json")
    checkpoint = Path(args.checkpoint or f"reports/{args.arch}_bci2a_{args.label_mode}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "history": history, "args": vars(args)},
        checkpoint,
    )

    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            data["x_all"],
            capture,
            layer_names=layer_names,
            device=device,
            dt=data["dt"],
            n_pairs=args.n_pairs,
            pair_batch_size=min(args.pair_batch_size, batch_size),
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            data["x_all"],
            capture,
            layer_names=layer_names,
            device=device,
            sfreq=sfreq,
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            data["x_all"],
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
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "labels_all": label_counts(data["labels_all"]),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "train_label_counts": {str(i): int((train_y == i).sum()) for i in sorted(set(train_y))},
            "val_label_counts": {str(i): int((val_y == i).sum()) for i in sorted(set(val_y))},
            "train_sessions": [
                s["name"]
                for s in data["sessions"]
                if s["name"].endswith("T") and s["name"] != "A09T"
            ],
            "val_sessions": ["A09T"],
            "audit_sessions": [s["name"] for s in data["sessions"]],
            "dt": data["dt"],
            "sfreq": sfreq,
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
    print(
        "final_train_acc",
        payload["training"]["final_train_acc"],
        "best_val_acc",
        f"{payload['training']['best_val_acc']:.3f}",
        flush=True,
    )
    for name in layer_names:
        print(
            name,
            "spearman=",
            f"{layers[name]['pairwise_isometry']['distance_spearman']:.4f}",
            "odi=",
            f"{layers[name]['band_orthogonality']['output_mean_abs_offdiag_cosine_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
