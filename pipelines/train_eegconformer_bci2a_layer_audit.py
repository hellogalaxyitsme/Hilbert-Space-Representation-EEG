#!/usr/bin/env python3
"""Train Braindecode EEGConformer on BCI IV 2a and audit HSDD layers."""

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
    LayerCapture,
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


EEGCONFORMER_LAYER_NAMES = (
    "patch_embedding.shallownet",
    "patch_embedding.projection",
    "patch_embedding",
    "transformer.0",
    "transformer.2",
    "transformer.5",
    "transformer",
    "fc",
    "final_layer",
    "logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default="reports/eegconformer_bci2a_layer_audit.json")
    parser.add_argument("--checkpoint", default="reports/eegconformer_bci2a.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-pairs", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--band-epochs", type=int, default=180)
    parser.add_argument("--local-points", type=int, default=12)
    parser.add_argument("--local-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def build_eegconformer(n_chans: int, n_times: int, n_outputs: int = 4) -> nn.Module:
    patch_braindecode_moabb_import()
    from braindecode.models import EEGConformer

    return EEGConformer(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        final_fc_length="auto",
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
    )


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
    model = build_eegconformer(n_chans=train_x.shape[1], n_times=train_x.shape[2])
    print(model, flush=True)
    model.to(device)

    if args.skip_training:
        val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
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
        print(
            f"skip_training val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
    else:
        history = train_model(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "args": vars(args),
            "layer_names": EEGCONFORMER_LAYER_NAMES,
        },
        checkpoint,
    )

    model.eval()
    capture = LayerCapture(model, EEGCONFORMER_LAYER_NAMES)
    try:
        pairwise = layer_pairwise_isometry(
            data["x_all"],
            capture,
            layer_names=EEGCONFORMER_LAYER_NAMES,
            device=device,
            dt=data["dt"],
            n_pairs=args.n_pairs,
            pair_batch_size=args.pair_batch_size,
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            data["x_all"],
            capture,
            layer_names=EEGCONFORMER_LAYER_NAMES,
            device=device,
            sfreq=1.0 / data["dt"],
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            data["x_all"],
            capture,
            layer_names=EEGCONFORMER_LAYER_NAMES,
            device=device,
            dt=data["dt"],
            n_points=args.local_points,
            n_directions=args.local_directions,
            eps=1e-3,
            seed=args.seed,
        )
    finally:
        capture.close()

    layer_reports = {}
    for name in EEGCONFORMER_LAYER_NAMES:
        layer_reports[name] = {
            "pairwise_isometry": pairwise[name],
            "band_orthogonality": band[name],
            "local_directional_distortion": local[name],
        }

    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "labels_all": label_counts(data["labels_all"]),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "train_label_counts": {
                str(i): int((train_y == i).sum()) for i in sorted(set(train_y))
            },
            "val_label_counts": {
                str(i): int((val_y == i).sum()) for i in sorted(set(val_y))
            },
            "train_sessions": [
                s["name"]
                for s in data["sessions"]
                if s["name"].endswith("T") and s["name"] != "A09T"
            ],
            "val_sessions": ["A09T"],
            "audit_sessions": [s["name"] for s in data["sessions"]],
            "dt": data["dt"],
            "sfreq": 1.0 / data["dt"],
        },
        "model": {
            "source": "braindecode.models.EEGConformer",
            "checkpoint": str(checkpoint),
            "layer_names": EEGCONFORMER_LAYER_NAMES,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "history": history,
            "best_val_acc": float(max(row["val_acc"] for row in history)),
            "final_val_acc": float(history[-1]["val_acc"]),
            "final_train_acc": (
                None
                if history[-1]["train_acc"] is None
                else float(history[-1]["train_acc"])
            ),
            "label_mode": args.label_mode,
            "skip_training": bool(args.skip_training),
        },
        "layers": layer_reports,
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}", flush=True)
    print(
        "final_train_acc",
        (
            "None"
            if payload["training"]["final_train_acc"] is None
            else f"{payload['training']['final_train_acc']:.3f}"
        ),
        "best_val_acc",
        f"{payload['training']['best_val_acc']:.3f}",
        flush=True,
    )
    for name in EEGCONFORMER_LAYER_NAMES:
        iso = layer_reports[name]["pairwise_isometry"]
        odi = layer_reports[name]["band_orthogonality"]
        print(
            name,
            "scaled_iso=",
            f"{iso['scaled_abs_ratio_error_mean']:.4f}",
            "spearman=",
            f"{iso['distance_spearman']:.4f}",
            "out_odi=",
            f"{odi['output_mean_abs_offdiag_cosine_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
