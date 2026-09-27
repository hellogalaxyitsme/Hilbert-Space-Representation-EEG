#!/usr/bin/env python3
"""Track EEGConformer HSDD metrics during training on BCI IV 2a."""

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
from train_eegconformer_bci2a_layer_audit import (
    EEGCONFORMER_LAYER_NAMES,
    build_eegconformer,
)
from train_eegnet_bci2a_dynamics import parse_checkpoints, train_one_epoch
from train_eegnet_bci2a_layer_audit import (
    LayerCapture,
    evaluate,
    label_counts,
    layer_band_orthogonality,
    layer_pairwise_isometry,
    load_bci2a,
    make_train_val,
    set_seed,
)


COMPACT_LAYERS = (
    "patch_embedding",
    "transformer.0",
    "transformer.2",
    "transformer.5",
    "fc",
    "logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default="reports/eegconformer_bci2a_dynamics_true.json")
    parser.add_argument("--checkpoint-dir", default="reports/eegconformer_dynamics_checkpoints")
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoints", default="0,1,2,5,10,20,30")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-pairs", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--band-epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def audit_checkpoint(
    model: nn.Module,
    x_all,
    *,
    device: torch.device,
    dt: float,
    n_pairs: int,
    pair_batch_size: int,
    band_epochs: int,
    seed: int,
) -> dict:
    model.eval()
    capture = LayerCapture(model, EEGCONFORMER_LAYER_NAMES)
    try:
        pairwise = layer_pairwise_isometry(
            x_all,
            capture,
            layer_names=EEGCONFORMER_LAYER_NAMES,
            device=device,
            dt=dt,
            n_pairs=n_pairs,
            pair_batch_size=pair_batch_size,
            seed=seed,
        )
        band = layer_band_orthogonality(
            x_all,
            capture,
            layer_names=EEGCONFORMER_LAYER_NAMES,
            device=device,
            sfreq=1.0 / dt,
            n_epochs=band_epochs,
            seed=seed,
        )
    finally:
        capture.close()

    return {
        name: {
            "pairwise_isometry": pairwise[name],
            "band_orthogonality": band[name],
        }
        for name in EEGCONFORMER_LAYER_NAMES
    }


def compact_layer_summary(layers: dict) -> dict:
    return {
        name: {
            "spearman": layers[name]["pairwise_isometry"]["distance_spearman"],
            "odi": layers[name]["band_orthogonality"][
                "output_mean_abs_offdiag_cosine_mean"
            ],
        }
        for name in COMPACT_LAYERS
    }


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, val_x, val_y = make_train_val(
        data,
        label_mode=args.label_mode,
        seed=args.seed,
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_eegconformer(n_chans=train_x.shape[1], n_times=train_x.shape[2]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    audits = []
    for epoch in range(0, args.epochs + 1):
        if epoch in checkpoints:
            train_loss, train_acc = evaluate(model, train_eval_loader, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            print(
                f"audit epoch {epoch:03d} label_mode={args.label_mode} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
                flush=True,
            )
            layers = audit_checkpoint(
                model,
                data["x_all"],
                device=device,
                dt=data["dt"],
                n_pairs=args.n_pairs,
                pair_batch_size=args.pair_batch_size,
                band_epochs=args.band_epochs,
                seed=args.seed,
            )
            audits.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "layers": layers,
                    "compact": compact_layer_summary(layers),
                }
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "label_mode": args.label_mode,
                    "args": vars(args),
                },
                checkpoint_dir / f"eegconformer_{args.label_mode}_epoch_{epoch:03d}.pt",
            )

        if epoch == args.epochs:
            break

        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch + 1,
            **train_metrics,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}",
            flush=True,
        )

    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "labels_all": label_counts(data["labels_all"]),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "dt": data["dt"],
            "sfreq": 1.0 / data["dt"],
        },
        "model": {
            "source": "braindecode.models.EEGConformer",
            "layer_names": EEGCONFORMER_LAYER_NAMES,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "run": {
            "label_mode": args.label_mode,
            "epochs": args.epochs,
            "checkpoints": checkpoints,
            "n_pairs": args.n_pairs,
            "band_epochs": args.band_epochs,
            "seed": args.seed,
        },
        "history": history,
        "audits": audits,
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
