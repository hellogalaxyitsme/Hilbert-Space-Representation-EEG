#!/usr/bin/env python3
"""Track HSDD dynamics for one Braindecode architecture on BCI IV 2a."""

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
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoints", default="0,1,2,5,10,20,30")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-pairs", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def compact_layers(arch: str, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    if arch == "eegnet":
        selected = (
            "conv_temporal",
            "conv_spatial",
            "bnorm_1",
            "conv_separable_depth",
            "bnorm_2",
            "pool_2",
            "logits",
        )
    elif arch == "shallowconvnet":
        selected = (
            "conv_time_spat",
            "bnorm",
            "conv_nonlin_exp",
            "pool",
            "pool_nonlin_exp",
            "logits",
        )
    elif arch == "eegconformer":
        selected = (
            "patch_embedding",
            "transformer.0",
            "transformer.2",
            "transformer.5",
            "fc",
            "logits",
        )
    elif arch == "tsception":
        selected = (
            "temporal_blocks.0",
            "temporal_blocks.2",
            "batch_temporal_lay",
            "spatial_block_1",
            "spatial_block_2",
            "batch_spatial_lay",
            "dense_layer",
            "logits",
        )
    elif arch == "atcnet":
        selected = (
            "conv_block",
            "attention_blocks.0",
            "attention_blocks.4",
            "temporal_conv_nets.0",
            "temporal_conv_nets.4",
            "final_layer.4",
            "logits",
        )
    else:
        selected = layer_names
    return tuple(name for name in selected if name in layer_names)


def audit_checkpoint(
    model,
    x,
    *,
    layer_names,
    device,
    dt,
    sfreq,
    n_pairs,
    pair_batch_size,
    band_epochs,
    seed,
):
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
    return {
        name: {"pairwise_isometry": pairwise[name], "band_orthogonality": band[name]}
        for name in layer_names
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
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=batch_size,
        shuffle=False,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(
        args.checkpoint_dir or f"reports/{args.arch}_bci2a_dynamics_{args.label_mode}_checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    audits = []
    compact = compact_layers(args.arch, layer_names)
    for epoch in range(args.epochs + 1):
        if epoch in checkpoints:
            train_loss, train_acc = evaluate(model, train_eval_loader, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            print(
                f"audit epoch {epoch:03d} arch={args.arch} labels={args.label_mode} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
                flush=True,
            )
            layers = audit_checkpoint(
                model,
                data["x_all"],
                layer_names=layer_names,
                device=device,
                dt=data["dt"],
                sfreq=sfreq,
                n_pairs=args.n_pairs,
                pair_batch_size=min(args.pair_batch_size, batch_size),
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
                    "compact": {
                        name: {
                            "spearman": layers[name]["pairwise_isometry"]["distance_spearman"],
                            "odi": layers[name]["band_orthogonality"]["output_mean_abs_offdiag_cosine_mean"],
                        }
                        for name in compact
                    },
                }
            )
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)},
                checkpoint_dir / f"{args.arch}_{args.label_mode}_epoch_{epoch:03d}.pt",
            )
        if epoch == args.epochs:
            break
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch + 1, **train_metrics, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(
            f"epoch {epoch + 1:03d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )

    out = Path(args.out or f"reports/{args.arch}_bci2a_dynamics_{args.label_mode}.json")
    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": args.root,
            "x_all_shape": list(data["x_all"].shape),
            "labels_all": label_counts(data["labels_all"]),
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "dt": data["dt"],
            "sfreq": sfreq,
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "layer_names": layer_names,
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
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
