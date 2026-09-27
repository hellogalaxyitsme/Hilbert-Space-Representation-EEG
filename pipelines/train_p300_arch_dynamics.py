#!/usr/bin/env python3
"""Track HSDD dynamics for one architecture on raw BNCI2014-009 P300 epochs."""

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
from p300_utils import label_counts, load_p300_cache
from train_bci2a_arch_dynamics import audit_checkpoint, compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_dynamics import parse_checkpoints, train_one_epoch
from train_eegnet_bci2a_layer_audit import evaluate, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--cache", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoints", default="0,1,2,5,10,20")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-subjects", default="9,10")
    parser.add_argument("--max-train-per-class", type=int, default=288)
    parser.add_argument("--max-val-per-class", type=int, default=72)
    parser.add_argument("--n-pairs", type=int, default=1024)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_p300_cache(
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
    requested_layer_names = layer_names
    layer_names = available_layer_names(model, layer_names)
    dropped_layer_names = [name for name in requested_layer_names if name not in layer_names]
    if dropped_layer_names:
        print(f"dropped unavailable audit layers: {dropped_layer_names}", flush=True)
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(train_y)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])), batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir or f"reports/p300_{args.arch}_dynamics_{args.label_mode}_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    audits = []
    compact = compact_layers(args.arch, layer_names)
    for epoch in range(args.epochs + 1):
        if epoch in checkpoints:
            train_loss, train_acc = evaluate(model, train_eval_loader, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            print(f"audit epoch {epoch:03d} arch={args.arch} labels={args.label_mode} train_acc={train_acc:.3f} val_acc={val_acc:.3f}", flush=True)
            layers = audit_checkpoint(
                model,
                data["x"],
                layer_names=layer_names,
                device=device,
                dt=data["dt"],
                sfreq=data["sfreq"],
                n_pairs=args.n_pairs,
                pair_batch_size=min(args.pair_batch_size, batch_size),
                band_epochs=args.band_epochs,
                seed=args.seed,
            )
            audits.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "layers": layers,
                "compact": {
                    name: {
                        "spearman": layers[name]["pairwise_isometry"]["distance_spearman"],
                        "odi": layers[name]["band_orthogonality"]["output_mean_abs_offdiag_cosine_mean"],
                    }
                    for name in compact
                },
            })
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)}, checkpoint_dir / f"{args.arch}_{args.label_mode}_epoch_{epoch:03d}.pt")
        if epoch == args.epochs:
            break
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch + 1, **train_metrics, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(f"epoch {epoch + 1:03d} train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)

    out = Path(args.out or f"reports/p300_{args.arch}_dynamics_{args.label_mode}.json")
    payload = {
        "dataset": {
            "kind": "BNCI2014_009_P300_raw_event_cache",
            "cache": args.cache,
            "x_shape": list(data["x"].shape),
            "sfreq": data["sfreq"],
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
            "requested_layer_names": requested_layer_names,
            "dropped_layer_names": dropped_layer_names,
            "layer_names": layer_names,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "run": {"label_mode": args.label_mode, "epochs": args.epochs, "checkpoints": checkpoints, "n_pairs": args.n_pairs, "band_epochs": args.band_epochs, "seed": args.seed},
        "history": history,
        "audits": audits,
    }
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
