#!/usr/bin/env python3
"""Train Braindecode EEGNet on BCI IV 2a and audit layer-wise HSDD metrics."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsdd.diagnostics import _spearman, cosine_matrix, l2_norm, offdiag_mean_abs
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components


EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "feet": 771,
    "tongue": 772,
}
UNKNOWN_EVAL_CODE = 783
LABEL_MAP = {769: 0, 770: 1, 771: 2, 772: 3}


LAYER_NAMES = (
    "conv_temporal",
    "bnorm_temporal",
    "conv_spatial",
    "bnorm_1",
    "elu_1",
    "pool_1",
    "conv_separable_depth",
    "conv_separable_point",
    "bnorm_2",
    "elu_2",
    "pool_2",
    "final_layer.conv_classifier",
    "logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default="reports/eegnet_bci2a_layer_audit.json")
    parser.add_argument("--checkpoint", default="reports/eegnet_bci2a.pt")
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
    parser.add_argument(
        "--label-mode",
        choices=("true", "shuffled"),
        default="true",
        help="Use true training labels or a fixed shuffled-label control.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Audit the initialized EEGNet without training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_braindecode_moabb_import() -> None:
    """Compatibility shim for Braindecode 1.2.0 with newer MOABB."""

    try:
        import moabb.datasets as md

        if not hasattr(md, "BNCI2014001") and hasattr(md, "BNCI2014_001"):
            md.BNCI2014001 = md.BNCI2014_001
    except Exception:
        pass


def discover_sessions(root: Path) -> list[Path]:
    sessions = []
    for subject in range(1, 10):
        for split in ("T", "E"):
            session = root / f"A{subject:02d}{split}"
            if session.exists():
                sessions.append(session)
    return sessions


def load_session(
    session_dir: Path,
    *,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    import mne

    raw_path = session_dir / f"{session_dir.name}_raw_cleaned.fif"
    if not raw_path.exists():
        raw_path = session_dir / f"{session_dir.name}_raw.fif"
    events_path = session_dir / f"{session_dir.name}_events.npy"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
    events = np.load(events_path)
    known = set(EVENT_ID.values())
    events_known = events[np.asarray([event[2] in known for event in events])]
    if len(events_known) == 0:
        events_known = events[np.asarray([event[2] == UNKNOWN_EVAL_CODE for event in events])]
        event_id = {"unknown_eval_cue": UNKNOWN_EVAL_CODE}
    else:
        event_id = EVENT_ID

    epochs = mne.Epochs(
        raw,
        events_known,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    x = epochs.get_data(copy=True).astype(np.float32)
    labels = epochs.events[:, 2].astype(int)
    x -= x.mean(axis=-1, keepdims=True)
    scale = np.std(x, axis=(-2, -1), keepdims=True)
    x = x / np.maximum(scale, 1e-6)
    return x, labels, 1.0 / float(raw.info["sfreq"]), [raw.ch_names[i] for i in picks]


def load_bci2a(root: Path, *, tmin: float, tmax: float) -> dict:
    sessions = []
    for session in discover_sessions(root):
        x, labels, dt, ch_names = load_session(session, tmin=tmin, tmax=tmax)
        sessions.append(
            {
                "name": session.name,
                "x": x,
                "labels": labels,
                "dt": dt,
                "ch_names": ch_names,
                "split": session.name[-1],
                "subject": session.name[:3],
            }
        )
        print("loaded", session.name, x.shape, label_counts(labels), flush=True)

    x_all = np.concatenate([s["x"] for s in sessions], axis=0)
    labels_all = np.concatenate([s["labels"] for s in sessions], axis=0)
    session_index = np.concatenate(
        [np.full(len(s["x"]), i, dtype=int) for i, s in enumerate(sessions)]
    )
    return {
        "sessions": sessions,
        "x_all": x_all,
        "labels_all": labels_all,
        "session_index": session_index,
        "dt": sessions[0]["dt"],
        "ch_names": sessions[0]["ch_names"],
    }


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(label): int((labels == label).sum()) for label in sorted(set(labels))}


def make_train_val(
    data: dict,
    *,
    label_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for session in data["sessions"]:
        labels = session["labels"]
        known_mask = np.asarray([int(label) in LABEL_MAP for label in labels])
        if not np.any(known_mask):
            continue
        y = np.asarray([LABEL_MAP[int(label)] for label in labels[known_mask]], dtype=np.int64)
        if session["name"] == "A09T":
            val_x.append(session["x"][known_mask])
            val_y.append(y)
        else:
            train_x.append(session["x"][known_mask])
            train_y.append(y)
    train_x_arr = np.concatenate(train_x, axis=0)
    train_y_arr = np.concatenate(train_y, axis=0)
    val_x_arr = np.concatenate(val_x, axis=0)
    val_y_arr = np.concatenate(val_y, axis=0)

    if label_mode == "shuffled":
        rng = np.random.default_rng(seed)
        train_y_arr = rng.permutation(train_y_arr)
    elif label_mode != "true":
        raise ValueError(f"unknown label_mode: {label_mode}")

    return (
        train_x_arr,
        train_y_arr,
        val_x_arr,
        val_y_arr,
    )


def build_eegnet(n_chans: int, n_times: int, n_outputs: int = 4) -> nn.Module:
    patch_braindecode_moabb_import()
    from braindecode.models import EEGNet

    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        final_conv_length="auto",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        depthwise_kernel_length=16,
        drop_prob=0.25,
    )


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(xb)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": total_correct / total,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}",
            flush=True,
        )
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item()) * len(xb)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return total_loss / total, total_correct / total


class LayerCapture:
    def __init__(self, model: nn.Module, layer_names: tuple[str, ...]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.outputs: dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        self.handles = []
        for name in layer_names:
            if name == "logits":
                continue
            if name not in modules:
                raise KeyError(f"missing layer {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            self.outputs[name] = output.detach()

        return hook

    @torch.no_grad()
    def __call__(self, x: np.ndarray, *, device: torch.device) -> dict[str, np.ndarray]:
        self.outputs = {}
        xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        logits = self.model(xb)
        captured = {name: tensor.detach().cpu().numpy() for name, tensor in self.outputs.items()}
        captured["logits"] = logits.detach().cpu().numpy()
        return captured

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def layer_pairwise_isometry(
    x: np.ndarray,
    capture: LayerCapture,
    *,
    layer_names: tuple[str, ...] = LAYER_NAMES,
    device: torch.device,
    dt: float,
    n_pairs: int,
    pair_batch_size: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]

    dx = x[pairs[:, 0]].reshape(n_pairs, -1) - x[pairs[:, 1]].reshape(n_pairs, -1)
    d_in = l2_norm(dx, dt=dt, axis=1)
    d_out_by_layer = {name: [] for name in layer_names}

    for start in range(0, n_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, n_pairs)
        idx0 = pairs[start:stop, 0]
        idx1 = pairs[start:stop, 1]
        batch = np.concatenate([x[idx0], x[idx1]], axis=0)
        outputs = capture(batch, device=device)
        b = stop - start
        for name in layer_names:
            z = outputs[name].reshape(2 * b, -1)
            d_out = np.linalg.norm(z[:b] - z[b:], axis=1)
            d_out_by_layer[name].append(d_out.astype(np.float64))

    reports = {}
    for name in layer_names:
        d_out = np.concatenate(d_out_by_layer[name])
        best_scale = float(np.dot(d_in, d_out) / max(np.dot(d_in, d_in), 1e-12))
        raw_ratios = d_out / np.maximum(d_in, 1e-12)
        scaled_ratios = d_out / np.maximum(best_scale * d_in, 1e-12)
        reports[name] = {
            "n_pairs": int(n_pairs),
            "best_scale": best_scale,
            "input_distance_mean": float(np.mean(d_in)),
            "output_distance_mean": float(np.mean(d_out)),
            "raw_abs_ratio_error_mean": float(np.mean(np.abs(raw_ratios - 1.0))),
            "scaled_abs_ratio_error_mean": float(np.mean(np.abs(scaled_ratios - 1.0))),
            "scaled_abs_ratio_error_median": float(np.median(np.abs(scaled_ratios - 1.0))),
            "distance_spearman": _spearman(d_in, d_out),
        }
    return reports


def layer_band_orthogonality(
    x: np.ndarray,
    capture: LayerCapture,
    *,
    layer_names: tuple[str, ...] = LAYER_NAMES,
    device: torch.device,
    sfreq: float,
    n_epochs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_epochs, len(x)), replace=False)
    per_layer = {
        name: {"input": [], "output": [], "input_matrix": [], "output_matrix": []}
        for name in layer_names
    }
    band_names = None
    for idx in indices:
        components, names = fft_band_components(x[idx], sfreq=sfreq)
        band_names = names
        input_cos = cosine_matrix(components)
        outputs = capture(components.astype(np.float32), device=device)
        for name in layer_names:
            output_cos = cosine_matrix(outputs[name])
            per_layer[name]["input"].append(offdiag_mean_abs(input_cos))
            per_layer[name]["output"].append(offdiag_mean_abs(output_cos))
            per_layer[name]["input_matrix"].append(input_cos)
            per_layer[name]["output_matrix"].append(output_cos)

    reports = {}
    for name, values in per_layer.items():
        reports[name] = {
            "band_names": band_names,
            "n_epochs": int(len(indices)),
            "input_mean_abs_offdiag_cosine_mean": float(np.mean(values["input"])),
            "output_mean_abs_offdiag_cosine_mean": float(np.mean(values["output"])),
            "output_mean_abs_offdiag_cosine_std": float(np.std(values["output"])),
            "collapse_factor_mean": float(
                np.mean(values["output"]) / max(float(np.mean(values["input"])), 1e-12)
            ),
            "input_cosine_matrix_mean": np.mean(values["input_matrix"], axis=0),
            "output_cosine_matrix_mean": np.mean(values["output_matrix"], axis=0),
        }
    return reports


def layer_local_directional_distortion(
    x: np.ndarray,
    capture: LayerCapture,
    *,
    layer_names: tuple[str, ...] = LAYER_NAMES,
    device: torch.device,
    dt: float,
    n_points: int,
    n_directions: int,
    eps: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_points, len(x)), replace=False)
    stretches = {name: [] for name in layer_names}
    for idx in indices:
        x0 = x[idx : idx + 1].astype(np.float32)
        directions = rng.normal(size=(n_directions,) + x0.shape[1:]).astype(np.float32)
        norms = l2_norm(directions.reshape(n_directions, -1), dt=dt, axis=1)
        directions = directions / np.maximum(norms[:, None, None], 1e-12)
        batch = np.concatenate([x0, x0 + eps * directions], axis=0)
        outputs = capture(batch.astype(np.float32), device=device)
        for name in layer_names:
            z = outputs[name].reshape(1 + n_directions, -1)
            local = np.linalg.norm(z[1:] - z[:1], axis=1) / eps
            stretches[name].extend(local.astype(float).tolist())

    reports = {}
    for name, vals in stretches.items():
        arr = np.asarray(vals, dtype=np.float64)
        reports[name] = {
            "n_points": int(len(indices)),
            "n_directions": int(n_directions),
            "eps": float(eps),
            "stretch_mean": float(np.mean(arr)),
            "stretch_median": float(np.median(arr)),
            "stretch_std": float(np.std(arr)),
            "stretch_min": float(np.min(arr)),
            "stretch_max": float(np.max(arr)),
        }
    return reports


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
    model = build_eegnet(n_chans=train_x.shape[1], n_times=train_x.shape[2])
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
        },
        checkpoint,
    )

    model.eval()
    capture = LayerCapture(model, LAYER_NAMES)
    try:
        pairwise = layer_pairwise_isometry(
            data["x_all"],
            capture,
            device=device,
            dt=data["dt"],
            n_pairs=args.n_pairs,
            pair_batch_size=args.pair_batch_size,
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            data["x_all"],
            capture,
            device=device,
            sfreq=1.0 / data["dt"],
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            data["x_all"],
            capture,
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
    for name in LAYER_NAMES:
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
            "train_label_counts": {str(i): int((train_y == i).sum()) for i in sorted(set(train_y))},
            "val_label_counts": {str(i): int((val_y == i).sum()) for i in sorted(set(val_y))},
            "train_sessions": [s["name"] for s in data["sessions"] if s["name"].endswith("T") and s["name"] != "A09T"],
            "val_sessions": ["A09T"],
            "audit_sessions": [s["name"] for s in data["sessions"]],
            "dt": data["dt"],
            "sfreq": 1.0 / data["dt"],
        },
        "model": {
            "source": "braindecode.models.EEGNet",
            "braindecode_moabb_compat_shim": True,
            "checkpoint": str(checkpoint),
            "layer_names": LAYER_NAMES,
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
    for name in LAYER_NAMES:
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
