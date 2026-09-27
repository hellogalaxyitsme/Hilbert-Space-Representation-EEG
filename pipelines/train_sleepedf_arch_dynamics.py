#!/usr/bin/env python3
"""Track HSDD dynamics for one architecture on cached Sleep-EDF."""

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
from train_eegnet_bci2a_dynamics import parse_checkpoints, train_one_epoch
from train_eegnet_bci2a_layer_audit import (
    LayerCapture,
    evaluate,
    layer_band_orthogonality,
    layer_pairwise_isometry,
    set_seed,
)
from train_sleepedf_arch_layer_audit import ARCHES, arch_spec, label_counts, load_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--cache", default="prepared/sleepedf_sc20_fpzcz_pzoz.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoints", default="0,1,2,5,10,20")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-recordings", type=int, default=4)
    parser.add_argument("--max-train-per-class", type=int, default=1600)
    parser.add_argument("--max-val-per-class", type=int, default=400)
    parser.add_argument("--n-pairs", type=int, default=1024)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def compact_layers(arch: str, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    if arch == "eegnet":
        return tuple(name for name in ("conv_temporal", "conv_spatial", "conv_separable_depth", "bnorm_2", "pool_2", "logits") if name in layer_names)
    if arch == "shallowconvnet":
        return tuple(name for name in ("conv_time_spat", "bnorm", "conv_nonlin_exp", "pool", "pool_nonlin_exp", "logits") if name in layer_names)
    if arch == "eegconformer":
        return tuple(name for name in ("patch_embedding", "transformer.0", "transformer.5", "fc", "logits") if name in layer_names)
    if arch == "tsception":
        return tuple(name for name in ("temporal_blocks.0", "temporal_blocks.2", "batch_temporal_lay", "spatial_block_1", "spatial_block_2", "batch_spatial_lay", "dense_layer", "logits") if name in layer_names)
    if arch == "atcnet":
        return tuple(name for name in ("conv_block", "attention_blocks.0", "attention_blocks.4", "temporal_conv_nets.0", "temporal_conv_nets.4", "final_layer.4", "logits") if name in layer_names)
    return layer_names


def audit_checkpoint(model, x, *, layer_names, device, dt, sfreq, n_pairs, pair_batch_size, band_epochs, seed):
    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            x,
            capture,
            layer_names=layer_names,
            device=device,
            dt=dt,
            n_pairs=n_pairs,
            pair_batch_size=pair_batch_size,
            seed=seed,
        )
        band = layer_band_orthogonality(
            x,
            capture,
            layer_names=layer_names,
            device=device,
            sfreq=sfreq,
            n_epochs=band_epochs,
            seed=seed,
        )
    finally:
        capture.close()
    return {name: {"pairwise_isometry": pairwise[name], "band_orthogonality": band[name]} for name in layer_names}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)
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
    batch_size = args.batch_size if args.arch not in {"eegconformer", "tsception", "atcnet"} else min(args.batch_size, 32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])), batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir or f"reports/sleepedf_{args.arch}_dynamics_{args.label_mode}_checkpoints")
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
                dt=1.0 / data["sfreq"],
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

    out = Path(args.out or f"reports/sleepedf_{args.arch}_dynamics_{args.label_mode}.json")
    payload = {
        "dataset": {
            "kind": "Sleep_EDF_sleep_cassette_cache",
            "cache": args.cache,
            "x_shape": list(data["x"].shape),
            "label_names": data["label_names"],
            "label_counts": label_counts(data["y"], data["label_names"]),
            "train_shape": list(data["train_x"].shape),
            "val_shape": list(data["val_x"].shape),
            "train_label_counts": label_counts(data["train_y"], data["label_names"]),
            "val_label_counts": label_counts(data["val_y"], data["label_names"]),
            "sfreq": data["sfreq"],
        },
        "model": {"arch": args.arch, "source": model_source, "layer_names": layer_names, "n_parameters": int(sum(p.numel() for p in model.parameters()))},
        "run": {"label_mode": args.label_mode, "epochs": args.epochs, "checkpoints": checkpoints, "n_pairs": args.n_pairs, "band_epochs": args.band_epochs, "seed": args.seed},
        "history": history,
        "audits": audits,
    }
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
