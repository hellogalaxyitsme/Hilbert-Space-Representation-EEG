#!/usr/bin/env python3
"""Subject-level ODI dynamics for BCI IV 2a checkpoints."""

from __future__ import annotations

import argparse
import json
import math
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

from hsdd.diagnostics import cosine_matrix, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_dynamics import parse_checkpoints
from train_eegnet_bci2a_layer_audit import LABEL_MAP, LayerCapture, load_bci2a, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoints", default="0,1,2,5,10,20,30")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--max-epochs-per-subject", type=int, default=72)
    parser.add_argument("--layers", choices=("compact", "all"), default="compact")
    return parser.parse_args()


def no_suffix_seed41_path(path: Path, seed: int) -> Path:
    if seed == 41:
        alt = Path(str(path).replace("_seed41", ""))
        if alt.exists():
            return alt
    return path


def default_checkpoint_dir(arch: str, mode: str, seed: int) -> Path:
    path = Path(f"reports/{arch}_bci2a_dynamics_{mode}_seed{seed}_checkpoints")
    if seed == 41 and arch in {"eegnet", "shallowconvnet"}:
        legacy = Path(f"reports/{arch}_dynamics_{mode}")
        if legacy.exists():
            return legacy
    return no_suffix_seed41_path(path, seed)


def checkpoint_file(checkpoint_dir: Path, arch: str, mode: str, epoch: int) -> Path:
    return checkpoint_dir / f"{arch}_{mode}_epoch_{epoch:03d}.pt"


def known_subject_sessions(data: dict[str, Any]) -> list[dict[str, Any]]:
    subjects = []
    for session in data["sessions"]:
        if not session["name"].endswith("T"):
            continue
        labels = session["labels"]
        known_mask = np.asarray([int(label) in LABEL_MAP for label in labels])
        if not np.any(known_mask):
            continue
        y = np.asarray([LABEL_MAP[int(label)] for label in labels[known_mask]], dtype=np.int64)
        subjects.append(
            {
                "subject": session["subject"],
                "session": session["name"],
                "split": "heldout" if session["name"] == "A09T" else "train_subject",
                "x": session["x"][known_mask].astype(np.float32),
                "y": y,
            }
        )
    return subjects


@torch.no_grad()
def subject_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, *, device: torch.device, batch_size: int) -> dict[str, float | int]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item()) * len(xb)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return {
        "n_epochs": int(total),
        "loss": float(total_loss / total),
        "accuracy": float(total_correct / total),
    }


def layer_subject_odi(
    x: np.ndarray,
    capture: LayerCapture,
    *,
    layer_names: tuple[str, ...],
    device: torch.device,
    sfreq: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    if max_epochs > 0 and len(indices) > max_epochs:
        indices = rng.choice(indices, size=max_epochs, replace=False)
    indices = np.sort(indices)

    components = []
    input_odi = []
    band_names = None
    for idx in indices:
        comp, names = fft_band_components(x[idx], sfreq=sfreq)
        band_names = names
        components.append(comp.astype(np.float32))
        input_odi.append(offdiag_mean_abs(cosine_matrix(comp)))

    comp_arr = np.asarray(components, dtype=np.float32)
    n_epochs = comp_arr.shape[0]
    n_bands = comp_arr.shape[1]
    flat = comp_arr.reshape(n_epochs * n_bands, *comp_arr.shape[2:])

    chunks = {name: [] for name in layer_names}
    for start in range(0, len(flat), batch_size):
        outputs = capture(flat[start : start + batch_size], device=device)
        for name in layer_names:
            chunks[name].append(outputs[name].reshape(outputs[name].shape[0], -1).astype(np.float64))

    reports: dict[str, dict[str, Any]] = {}
    for name, parts in chunks.items():
        z = np.concatenate(parts, axis=0).reshape(n_epochs, n_bands, -1)
        vals = [offdiag_mean_abs(cosine_matrix(z[i])) for i in range(n_epochs)]
        reports[name] = {
            "band_names": band_names,
            "n_epochs": int(n_epochs),
            "input_mean_abs_offdiag_cosine_mean": float(np.mean(input_odi)),
            "output_mean_abs_offdiag_cosine_mean": float(np.mean(vals)),
            "output_mean_abs_offdiag_cosine_std": float(np.std(vals)),
            "collapse_factor_mean": float(np.mean(vals) / max(float(np.mean(input_odi)), 1e-12)),
        }
    return reports


def read_history_from_dynamics(arch: str, mode: str, seed: int) -> dict[int, dict[str, Any]]:
    if arch:
        path = Path(f"reports/{arch}_bci2a_dynamics_{mode}_seed{seed}.json")
        path = no_suffix_seed41_path(path, seed)
        if path.exists():
            payload = json.loads(path.read_text())
            return {int(row["epoch"]): row for row in payload.get("audits", [])}
    return {}


def available_layer_names(model: nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in layer_names if name == "logits" or name in modules)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    subjects = known_subject_sessions(data)
    if not subjects:
        raise RuntimeError("no labeled BCI IV 2a training sessions found")

    first_x = subjects[0]["x"]
    sfreq = 1.0 / float(data["dt"])
    model, layer_names, model_source = arch_spec(
        args.arch,
        first_x.shape[1],
        first_x.shape[2],
        n_outputs=4,
        sfreq=sfreq,
    )
    layer_names = available_layer_names(model, layer_names)
    if args.layers == "compact":
        layer_names = compact_layers(args.arch, layer_names)
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(args.arch, args.label_mode, args.seed)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"missing checkpoint dir: {checkpoint_dir}")

    global_history = read_history_from_dynamics(args.arch, args.label_mode, args.seed)
    audits = []
    for epoch in checkpoints:
        ckpt = checkpoint_file(checkpoint_dir, args.arch, args.label_mode, epoch)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        capture = LayerCapture(model.eval(), layer_names)
        try:
            subject_rows = []
            for si, subject in enumerate(subjects):
                acc = subject_accuracy(
                    model,
                    subject["x"],
                    subject["y"],
                    device=device,
                    batch_size=batch_size,
                )
                odi = layer_subject_odi(
                    subject["x"],
                    capture,
                    layer_names=layer_names,
                    device=device,
                    sfreq=sfreq,
                    max_epochs=args.max_epochs_per_subject,
                    batch_size=batch_size,
                    seed=args.seed * 1000 + epoch * 100 + si,
                )
                subject_rows.append(
                    {
                        "subject": subject["subject"],
                        "session": subject["session"],
                        "split": subject["split"],
                        **acc,
                        "layers": odi,
                    }
                )
        finally:
            capture.close()
        dyn = global_history.get(epoch, {})
        audits.append(
            {
                "epoch": int(epoch),
                "checkpoint": str(ckpt),
                "global_train_acc": None if dyn.get("train_acc") is None else float(dyn.get("train_acc")),
                "global_val_acc": None if dyn.get("val_acc") is None else float(dyn.get("val_acc")),
                "subjects": subject_rows,
            }
        )
        print(
            f"done arch={args.arch} mode={args.label_mode} seed={args.seed} epoch={epoch:03d}",
            flush=True,
        )

    payload = {
        "dataset": {
            "name": "bci2a",
            "root": args.root,
            "subject_sessions": [s["session"] for s in subjects],
            "heldout_session": "A09T",
            "dt": data["dt"],
            "sfreq": sfreq,
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "layer_names": layer_names,
        },
        "run": {
            "label_mode": args.label_mode,
            "seed": args.seed,
            "epochs": args.epochs,
            "checkpoints": checkpoints,
            "checkpoint_dir": str(checkpoint_dir),
            "max_epochs_per_subject": args.max_epochs_per_subject,
            "layers": args.layers,
        },
        "audits": audits,
    }
    out = Path(args.out or f"reports/bci2a_subject_odi/{args.arch}_{args.label_mode}_seed{args.seed}_subject_odi_dynamics.json")
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
