#!/usr/bin/env python3
"""Longer BCI IV 2a convergence runs for EEGConformer/TSception."""

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
from torch.utils.data import DataLoader, TensorDataset

from hsdd.io import write_json
from train_bci2a_arch_layer_audit import arch_spec, effective_batch_size
from train_eegnet_bci2a_dynamics import train_one_epoch
from train_eegnet_bci2a_layer_audit import evaluate, label_counts, load_bci2a, make_train_val, set_seed


ARCHES = ("eegconformer", "tsception")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true",), default="true")
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
    sfreq = 1.0 / data["dt"]
    model, layer_names, model_source = arch_spec(
        args.arch,
        train_x.shape[1],
        train_x.shape[2],
        n_outputs=4,
        sfreq=sfreq,
    )
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_loss_eval, train_acc_eval = evaluate(model, train_eval_loader, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            **train_metrics,
            "train_eval_loss": train_loss_eval,
            "train_eval_acc": train_acc_eval,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        improved = val_acc > best_val_acc + args.min_delta
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d} arch={args.arch} seed={args.seed} "
            f"train_acc={train_acc_eval:.3f} val_acc={val_acc:.3f} best={best_val_acc:.3f}@{best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} stale={stale}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_loss, final_train_acc = evaluate(model, train_eval_loader, loss_fn, device)
    final_val_loss, final_val_acc = evaluate(model, val_loader, loss_fn, device)
    checkpoint = Path(args.checkpoint or f"reports/tier2_convergence/{args.arch}_bci2a_convergence_true_seed{args.seed}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
        },
        checkpoint,
    )
    payload = {
        "dataset": {
            "name": "bci2a",
            "protocol": "cross-subject: A01T-A08T train, A09T validation/test proxy from preconverted BCI IV 2a FIF cache",
            "x_all_shape": list(data["x_all"].shape),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "train_label_counts": label_counts(train_y),
            "val_label_counts": label_counts(val_y),
            "tmin": args.tmin,
            "tmax": args.tmax,
            "sfreq": sfreq,
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "layer_names": layer_names,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "run": {
            "seed": args.seed,
            "label_mode": args.label_mode,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "checkpoint": str(checkpoint),
        },
        "summary": {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_train_acc_at_best": final_train_acc,
            "final_val_acc_at_best": final_val_acc,
            "stopped_epoch": history[-1]["epoch"] if history else 0,
        },
        "history": history,
    }
    out = Path(args.out or f"reports/tier2_convergence/{args.arch}_bci2a_convergence_true_seed{args.seed}.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
