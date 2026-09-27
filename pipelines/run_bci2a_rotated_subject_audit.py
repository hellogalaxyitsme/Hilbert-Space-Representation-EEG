#!/usr/bin/env python3
"""Train BCI IV 2a with a rotated held-out subject and run a static HSDD audit."""

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
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import (
    LABEL_MAP,
    LayerCapture,
    evaluate,
    label_counts,
    layer_band_orthogonality,
    layer_local_directional_distortion,
    layer_pairwise_isometry,
    load_bci2a,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--heldout-subject", required=True, help="Subject id such as A01 or A09.")
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
    parser.add_argument("--n-pairs", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--band-epochs", type=int, default=120)
    parser.add_argument("--local-points", type=int, default=8)
    parser.add_argument("--local-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def normalize_subject(subject: str) -> str:
    s = subject.strip().upper()
    if s.startswith("A") and len(s) == 3:
        return s
    if s.isdigit():
        return f"A{int(s):02d}"
    raise ValueError(f"invalid BCI subject id: {subject}")


def labelled_session_arrays(session: dict) -> tuple[np.ndarray, np.ndarray]:
    labels = session["labels"]
    known_mask = np.asarray([int(label) in LABEL_MAP for label in labels])
    y = np.asarray([LABEL_MAP[int(label)] for label in labels[known_mask]], dtype=np.int64)
    return session["x"][known_mask], y


def make_rotated_split(
    data: dict,
    *,
    heldout_subject: str,
    label_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray, np.ndarray]:
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    train_sessions: list[str] = []
    val_sessions: list[str] = []
    audit_x: list[np.ndarray] = []
    audit_y: list[np.ndarray] = []
    for session in data["sessions"]:
        if session["split"] != "T":
            continue
        x, y = labelled_session_arrays(session)
        if len(y) == 0:
            continue
        audit_x.append(x)
        audit_y.append(y)
        if session["subject"] == heldout_subject:
            val_x.append(x)
            val_y.append(y)
            val_sessions.append(session["name"])
        else:
            train_x.append(x)
            train_y.append(y)
            train_sessions.append(session["name"])
    if not train_x or not val_x:
        raise ValueError(f"could not build split for heldout subject {heldout_subject}")
    train_x_arr = np.concatenate(train_x, axis=0)
    train_y_arr = np.concatenate(train_y, axis=0)
    val_x_arr = np.concatenate(val_x, axis=0)
    val_y_arr = np.concatenate(val_y, axis=0)
    audit_x_arr = np.concatenate(audit_x, axis=0)
    audit_y_arr = np.concatenate(audit_y, axis=0)
    if label_mode == "shuffled":
        rng = np.random.default_rng(seed)
        train_y_arr = rng.permutation(train_y_arr)
    elif label_mode != "true":
        raise ValueError(label_mode)
    return train_x_arr, train_y_arr, val_x_arr, val_y_arr, train_sessions, val_sessions, audit_x_arr, audit_y_arr


def main() -> None:
    args = parse_args()
    heldout_subject = normalize_subject(args.heldout_subject)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    (
        train_x,
        train_y,
        val_x,
        val_y,
        train_sessions,
        val_sessions,
        audit_x,
        audit_y,
    ) = make_rotated_split(
        data,
        heldout_subject=heldout_subject,
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

    out = Path(
        args.out
        or f"reports/bci2a_rotated_subject/{args.arch}_{heldout_subject}_{args.label_mode}_seed{args.seed}.json"
    )
    checkpoint = Path(
        args.checkpoint
        or f"reports/bci2a_rotated_subject/{args.arch}_{heldout_subject}_{args.label_mode}_seed{args.seed}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "history": history, "args": vars(args)},
        checkpoint,
    )

    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            audit_x,
            capture,
            layer_names=layer_names,
            device=device,
            dt=data["dt"],
            n_pairs=args.n_pairs,
            pair_batch_size=min(args.pair_batch_size, batch_size),
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            audit_x,
            capture,
            layer_names=layer_names,
            device=device,
            sfreq=sfreq,
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            audit_x,
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
            "kind": "BCI_IV_2a_rotated_subject_split",
            "root": args.root,
            "heldout_subject": heldout_subject,
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "audit_sessions": [s["name"] for s in data["sessions"] if s["split"] == "T"],
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "audit_shape": list(audit_x.shape),
            "train_label_counts": {str(i): int((train_y == i).sum()) for i in sorted(set(train_y))},
            "val_label_counts": {str(i): int((val_y == i).sum()) for i in sorted(set(val_y))},
            "audit_label_counts": {str(i): int((audit_y == i).sum()) for i in sorted(set(audit_y))},
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
    print(
        "wrote",
        out,
        "heldout",
        heldout_subject,
        "arch",
        args.arch,
        "mode",
        args.label_mode,
        "seed",
        args.seed,
        "best_val_acc",
        f"{payload['training']['best_val_acc']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
