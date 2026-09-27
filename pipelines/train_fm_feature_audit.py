#!/usr/bin/env python3
"""Frozen EEG foundation-model feature audit with a linear probe head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
for path in (VENDOR_ROOT, PROJECT_ROOT, SCRIPTS_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mne
import numpy as np
import torch
from scipy.signal import resample
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsdd.io import write_json
from train_eegnet_bci2a_layer_audit import (
    LayerCapture,
    label_counts as bci_label_counts,
    layer_band_orthogonality,
    layer_local_directional_distortion,
    layer_pairwise_isometry,
    load_bci2a,
    make_train_val,
    set_seed,
)
from train_sleepedf_arch_layer_audit import (
    balanced_indices,
    label_counts as named_label_counts,
    load_cache,
)
from seediv_utils import load_seediv_cache
from p300_utils import load_p300_cache


FM_SPECS: dict[str, dict[str, Any]] = {
    "biot": {
        "class_name": "BIOT",
        "repo": "braindecode/biot-pretrained-prest-16chs",
        "sfreq": 200.0,
        "n_times": 1000,
        "n_chans": 16,
        "feature_dim": 256,
        "source": "braindecode.models.BIOT.from_pretrained",
    },
    "labram": {
        "class_name": "InterpolatedLaBraM",
        "repo": "braindecode/labram-pretrained",
        "sfreq": 200.0,
        "n_times": 3000,
        "feature_dim": 200,
        "source": "braindecode.models.InterpolatedLaBraM.from_pretrained",
    },
    "reve": {
        "class_name": "REVE",
        "repo": "brain-bzh/reve-base",
        "sfreq": 200.0,
        "n_times": 800,
        "feature_dim": 512,
        "source": "braindecode.models.REVE.from_pretrained",
    },
    "eegpt": {
        "class_name": "EEGPT",
        "repo": "braindecode/eegpt-pretrained",
        "sfreq": 250.0,
        "n_times": 1000,
        "n_chans": 62,
        "feature_dim": 2048,
        "source": "braindecode.models.EEGPT.from_pretrained",
    },
}

BCI_CH_NAMES = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)

BIOT_16CH = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "C3",
    "C4",
    "T7",
    "T8",
    "P3",
    "P4",
    "P7",
    "P8",
    "O1",
    "O2",
)

EEGPT_62CH = (
    "FP1",
    "FPZ",
    "FP2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "O1",
    "OZ",
    "O2",
)


def clean_ch_name(name: str) -> str:
    name = name.replace("EEG ", "").replace("EEG-", "").strip()
    name = name.replace("FP", "Fp").replace("Z", "z")
    return name


def montage_positions() -> dict[str, np.ndarray]:
    pos: dict[str, np.ndarray] = {}
    for montage_name in ("standard_1005", "standard_1020"):
        montage = mne.channels.make_standard_montage(montage_name)
        for key, value in montage.get_positions()["ch_pos"].items():
            pos[key.upper()] = np.asarray(value, dtype=np.float32)
    return pos


def position_for_name(name: str, pos: dict[str, np.ndarray]) -> np.ndarray | None:
    clean = clean_ch_name(name)
    if "-" in clean:
        parts = [part.strip() for part in clean.split("-") if part.strip()]
        coords = [pos.get(part.upper()) for part in parts]
        coords = [coord for coord in coords if coord is not None]
        if coords:
            return np.mean(np.stack(coords), axis=0).astype(np.float32)
    return pos.get(clean.upper())


def make_chs_info(ch_names: list[str], sfreq: float) -> list[dict[str, Any]]:
    pos = montage_positions()
    chs_info = []
    for idx, name in enumerate(ch_names):
        loc = np.zeros(12, dtype=float)
        coord = position_for_name(name, pos)
        if coord is not None:
            loc[:3] = coord
        chs_info.append(
            {
                "ch_name": clean_ch_name(name),
                "kind": 2,
                "coil_type": 1,
                "unit": 107,
                "loc": loc,
                "cal": 1.0,
                "range": 1.0,
                "scanno": idx + 1,
                "logno": idx + 1,
                "unit_mul": 0,
                "coord_frame": 4,
            }
        )
    return chs_info


def ensure_reve_position_cache(ch_names: list[str]) -> None:
    cache = VENDOR_ROOT / "braindecode" / "models" / ".cache" / "reve_positions.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    names = set(BCI_CH_NAMES) | {clean_ch_name(name) for name in ch_names}
    pos = montage_positions()
    out = {}
    for name in sorted(names):
        coord = position_for_name(name, pos)
        if coord is not None:
            out[name] = [float(v) for v in coord]
    if not out:
        out = {"Cz": [0.0, 0.0, 0.08]}
    cache.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def fit_time_axis(x: np.ndarray, n_times: int) -> np.ndarray:
    if x.shape[-1] > n_times:
        return x[..., :n_times]
    if x.shape[-1] < n_times:
        return np.pad(x, [(0, 0), (0, 0), (0, n_times - x.shape[-1])], mode="constant")
    return x


def fit_sfreq(x: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return x
    n_times = int(round(x.shape[-1] * target_sfreq / source_sfreq))
    return resample(x, n_times, axis=-1).astype(np.float32, copy=False)


def inverse_distance_adapt(
    x: np.ndarray,
    ch_names: list[str],
    target_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    pos = montage_positions()
    source_pos = [position_for_name(name, pos) for name in ch_names]
    target_pos = [position_for_name(name, pos) for name in target_names]
    out = np.zeros((x.shape[0], len(target_names), x.shape[-1]), dtype=np.float32)
    valid_sources = [i for i, coord in enumerate(source_pos) if coord is not None]
    if not valid_sources:
        out[:] = x.mean(axis=1, keepdims=True)
        return out, list(target_names)
    for target_idx, coord in enumerate(target_pos):
        if coord is None:
            out[:, target_idx] = x[:, valid_sources].mean(axis=1)
            continue
        distances = np.asarray(
            [np.linalg.norm(coord - source_pos[i]) for i in valid_sources],
            dtype=np.float64,
        )
        exact = np.flatnonzero(distances < 1e-8)
        if len(exact):
            out[:, target_idx] = x[:, valid_sources[int(exact[0])]]
        else:
            weights = 1.0 / np.maximum(distances, 1e-4) ** 2
            weights /= weights.sum()
            out[:, target_idx] = np.einsum("c,nct->nt", weights, x[:, valid_sources])
    return out, list(target_names)


def eegpt_adapt(x: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    clean = [clean_ch_name(name).upper() for name in ch_names]
    out = np.zeros((x.shape[0], len(EEGPT_62CH), x.shape[-1]), dtype=np.float32)
    for src_idx, name in enumerate(clean):
        if name in EEGPT_62CH:
            out[:, EEGPT_62CH.index(name)] = x[:, src_idx]
    return out, list(EEGPT_62CH)


def keep_positioned_channels(x: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    pos = montage_positions()
    keep = []
    dropped = []
    for idx, name in enumerate(ch_names):
        if position_for_name(name, pos) is None:
            dropped.append(name)
        else:
            keep.append(idx)
    if not keep:
        return x, ch_names, dropped
    return x[:, keep], [ch_names[idx] for idx in keep], dropped


def prepare_fm_input(
    x: np.ndarray,
    ch_names: list[str],
    source_sfreq: float,
    fm_id: str,
) -> tuple[np.ndarray, list[str]]:
    spec = FM_SPECS[fm_id]
    x = fit_sfreq(x.astype(np.float32, copy=False), source_sfreq, spec["sfreq"])
    if fm_id == "biot":
        x, ch_names = inverse_distance_adapt(x, ch_names, BIOT_16CH)
    elif fm_id == "eegpt":
        x, ch_names = eegpt_adapt(x, ch_names)
    else:
        ch_names = [clean_ch_name(name) for name in ch_names]
        if fm_id == "labram":
            ch_names = [name.split("-")[0] if "-" in name else name for name in ch_names]
            x, ch_names, dropped = keep_positioned_channels(x, ch_names)
            if dropped:
                print(f"LaBram dropped channels without standard positions: {dropped}", flush=True)
    x = fit_time_axis(x, spec["n_times"]).astype(np.float32, copy=False)
    return x, ch_names


def build_fm(fm_id: str, n_outputs: int, ch_names: list[str]) -> nn.Module:
    import braindecode.models as models

    spec = FM_SPECS[fm_id]
    if fm_id == "reve":
        ensure_reve_position_cache(ch_names)
    cls = getattr(models, spec["class_name"])
    common = {
        "n_outputs": n_outputs,
        "n_times": spec["n_times"],
        "sfreq": spec["sfreq"],
    }
    if fm_id == "biot":
        return cls.from_pretrained(
            spec["repo"],
            n_chans=spec["n_chans"],
            return_feature=True,
            **common,
        )
    if fm_id == "labram":
        labram_cls = cls
        if len(ch_names) < 4:
            labram_cls = getattr(models, "Labram")
        return labram_cls.from_pretrained(
            spec["repo"],
            **(
                {
                    "chs_info": make_chs_info(ch_names, spec["sfreq"]),
                    "n_chans": len(ch_names),
                }
                if len(ch_names) >= 4
                else {}
            ),
            **common,
        )
    if fm_id == "eegpt":
        return cls.from_pretrained(
            spec["repo"],
            n_chans=spec["n_chans"],
            chan_proj_type="none",
            **common,
        )
    if fm_id == "reve":
        return cls.from_pretrained(
            spec["repo"],
            n_chans=len(ch_names),
            **common,
        )
    raise ValueError(fm_id)


class FrozenFMLinearProbe(nn.Module):
    def __init__(self, fm_id: str, fm: nn.Module, feature_dim: int, n_outputs: int, ch_names: list[str]):
        super().__init__()
        self.fm_id = fm_id
        self.fm = fm.eval()
        for param in self.fm.parameters():
            param.requires_grad = False
        self.feature_identity = nn.Identity()
        self.head = nn.Linear(feature_dim, n_outputs)
        self.ch_names = ch_names
        self.register_buffer("reve_pos", torch.empty(0), persistent=False)
        if fm_id == "reve":
            pos = montage_positions()
            coords = []
            for name in ch_names:
                coord = position_for_name(name, pos)
                if coord is None:
                    coord = np.zeros(3, dtype=np.float32)
                coords.append(coord)
            self.reve_pos = torch.tensor(np.stack(coords), dtype=torch.float32)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.fm_id == "biot":
                _logits, features = self.fm(x)
            elif self.fm_id == "labram":
                try:
                    out = self.fm(x, return_features=True)
                except (TypeError, ValueError):
                    out = self.fm(
                        x,
                        ch_names=[name.upper() for name in self.ch_names],
                        return_features=True,
                    )
                features = out["cls_token"]
            elif self.fm_id == "eegpt":
                out = self.fm(x, return_features=True)
                features = out["features"]
                if features.ndim == 4:
                    features = features.mean(dim=(1, 2))
                elif features.ndim == 3:
                    features = features.mean(dim=1)
            elif self.fm_id == "reve":
                pos = self.reve_pos.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
                out = self.fm(x, pos=pos, return_features=True)
                features = out["features"]
                if features.ndim == 4:
                    features = features.mean(dim=(1, 2))
                elif features.ndim == 3:
                    features = features.mean(dim=1)
            else:
                raise ValueError(self.fm_id)
        return features.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_identity(self.extract_features(x))
        return self.head(features)


@torch.no_grad()
def extract_feature_array(
    model: FrozenFMLinearProbe,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval().to(device)
    feats = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        feats.append(model.extract_features(xb).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


def train_linear_head(
    model: FrozenFMLinearProbe,
    train_features: np.ndarray,
    train_y: np.ndarray,
    val_features: np.ndarray,
    val_y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    head = model.head.to(device)
    train_ds = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            total_correct += int((logits.argmax(1) == yb).sum().item())
            total += len(xb)
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = head(xb)
                val_loss += float(loss_fn(logits, yb).item()) * len(xb)
                val_correct += int((logits.argmax(1) == yb).sum().item())
                val_total += len(xb)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": total_correct / total,
            "val_loss": val_loss / val_total,
            "val_acc": val_correct / val_total,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_acc={row['train_acc']:.3f} val_acc={row['val_acc']:.3f}",
            flush=True,
        )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bci2a", "sleepedf", "seediv", "p300"), required=True)
    parser.add_argument("--fm", choices=tuple(FM_SPECS), required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fm-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bci-root", default=None)
    parser.add_argument("--bci-tmin", type=float, default=0.0)
    parser.add_argument("--bci-tmax", type=float, default=4.0)
    parser.add_argument("--sleep-cache", default="prepared/sleepedf_sc20_fpzcz_pzoz.npz")
    parser.add_argument("--sleep-val-recordings", type=int, default=4)
    parser.add_argument("--sleep-max-train-per-class", type=int, default=800)
    parser.add_argument("--sleep-max-val-per-class", type=int, default=200)
    parser.add_argument("--seediv-cache", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--seediv-val-subjects", default="13,14,15")
    parser.add_argument("--seediv-max-train-per-class", type=int, default=800)
    parser.add_argument("--seediv-max-val-per-class", type=int, default=200)
    parser.add_argument("--p300-cache", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--p300-val-subjects", default="9,10")
    parser.add_argument("--p300-max-train-per-class", type=int, default=288)
    parser.add_argument("--p300-max-val-per-class", type=int, default=72)
    parser.add_argument("--max-audit-epochs", type=int, default=2048)
    parser.add_argument("--n-pairs", type=int, default=1024)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--band-epochs", type=int, default=64)
    parser.add_argument("--local-points", type=int, default=6)
    parser.add_argument("--local-directions", type=int, default=3)
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci2a":
        data = load_bci2a(Path(args.bci_root), tmin=args.bci_tmin, tmax=args.bci_tmax)
        train_x, train_y, val_x, val_y = make_train_val(data, label_mode=args.label_mode, seed=args.seed)
        return {
            "x_all": data["x_all"],
            "y_all": data["labels_all"],
            "source_sfreq": 1.0 / data["dt"],
            "ch_names": data["ch_names"],
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "label_names": ["left_hand", "right_hand", "feet", "tongue"],
            "dataset_meta": {
                "kind": "BCI_IV_2a_preconverted_fif_full_available",
                "root": args.bci_root,
                "x_all_shape": list(data["x_all"].shape),
                "labels_all": bci_label_counts(data["labels_all"]),
                "dt": data["dt"],
                "sfreq": 1.0 / data["dt"],
                "ch_names": data["ch_names"],
                "tmin": args.bci_tmin,
                "tmax": args.bci_tmax,
            },
        }
    if args.dataset == "sleepedf":
        data = load_cache(
            Path(args.sleep_cache),
            args.sleep_val_recordings,
            args.seed,
            args.sleep_max_train_per_class,
            args.sleep_max_val_per_class,
        )
        train_y = data["train_y"].copy()
        if args.label_mode == "shuffled":
            train_y = np.random.default_rng(args.seed).permutation(train_y)
        return {
            "x_all": data["x"],
            "y_all": data["y"],
            "source_sfreq": data["sfreq"],
            "ch_names": data["channel_names"],
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "label_names": data["label_names"],
            "dataset_meta": {
                "kind": "Sleep_EDF_sleep_cassette_cache",
                "cache": args.sleep_cache,
                "x_shape": list(data["x"].shape),
                "sfreq": data["sfreq"],
                "channel_names": data["channel_names"],
                "label_names": data["label_names"],
                "label_counts": named_label_counts(data["y"], data["label_names"]),
                "train_recordings": data["train_recordings"],
                "val_recordings": data["val_recordings"],
            },
        }
    if args.dataset == "seediv":
        data = load_seediv_cache(
            Path(args.seediv_cache),
            val_subjects=args.seediv_val_subjects,
            seed=args.seed,
            max_train_per_class=args.seediv_max_train_per_class,
            max_val_per_class=args.seediv_max_val_per_class,
        )
        train_y = data["train_y"].copy()
        if args.label_mode == "shuffled":
            train_y = np.random.default_rng(args.seed).permutation(train_y)
        return {
            "x_all": data["x"],
            "y_all": data["y"],
            "source_sfreq": data["sfreq"],
            "ch_names": data["channel_names"],
            "train_x": data["train_x"],
            "train_y": train_y,
            "val_x": data["val_x"],
            "val_y": data["val_y"],
            "label_names": data["label_names"],
            "dataset_meta": {
                "kind": "SEED_IV_raw_eeg_window_cache",
                "cache": args.seediv_cache,
                "x_shape": list(data["x"].shape),
                "sfreq": data["sfreq"],
                "channel_names": data["channel_names"],
                "label_names": data["label_names"],
                "label_counts": named_label_counts(data["y"], data["label_names"]),
                "train_subjects": data["train_subjects"],
                "val_subjects": data["val_subjects"],
                "train_shape": list(data["train_x"].shape),
                "val_shape": list(data["val_x"].shape),
            },
        }
    data = load_p300_cache(
        Path(args.p300_cache),
        val_subjects=args.p300_val_subjects,
        seed=args.seed,
        max_train_per_class=args.p300_max_train_per_class,
        max_val_per_class=args.p300_max_val_per_class,
    )
    train_y = data["train_y"].copy()
    if args.label_mode == "shuffled":
        train_y = np.random.default_rng(args.seed).permutation(train_y)
    return {
        "x_all": data["x"],
        "y_all": data["y"],
        "source_sfreq": data["sfreq"],
        "ch_names": data["channel_names"],
        "train_x": data["train_x"],
        "train_y": train_y,
        "val_x": data["val_x"],
        "val_y": data["val_y"],
        "label_names": data["label_names"],
        "dataset_meta": {
            "kind": "BNCI2014_009_P300_raw_event_cache",
            "cache": args.p300_cache,
            "x_shape": list(data["x"].shape),
            "sfreq": data["sfreq"],
            "channel_names": data["channel_names"],
            "label_names": data["label_names"],
            "label_counts": named_label_counts(data["y"], data["label_names"]),
            "train_subjects": data["train_subjects"],
            "val_subjects": data["val_subjects"],
            "train_shape": list(data["train_x"].shape),
            "val_shape": list(data["val_x"].shape),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args)
    n_outputs = len(data["label_names"])
    spec = FM_SPECS[args.fm]

    train_x, adapted_ch_names = prepare_fm_input(data["train_x"], data["ch_names"], data["source_sfreq"], args.fm)
    val_x, _ = prepare_fm_input(data["val_x"], data["ch_names"], data["source_sfreq"], args.fm)
    audit_idx = np.arange(len(data["x_all"]))
    if args.max_audit_epochs > 0 and len(audit_idx) > args.max_audit_epochs:
        audit_idx = np.random.default_rng(args.seed).choice(audit_idx, size=args.max_audit_epochs, replace=False)
    audit_x, _ = prepare_fm_input(data["x_all"][audit_idx], data["ch_names"], data["source_sfreq"], args.fm)

    fm = build_fm(args.fm, n_outputs, adapted_ch_names)
    model = FrozenFMLinearProbe(args.fm, fm, spec["feature_dim"], n_outputs, adapted_ch_names).to(device)
    print(model, flush=True)

    train_features = extract_feature_array(model, train_x, device=device, batch_size=args.fm_batch_size)
    val_features = extract_feature_array(model, val_x, device=device, batch_size=args.fm_batch_size)
    if args.skip_training:
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = model.head(torch.from_numpy(val_features).to(device))
            val_loss = float(loss_fn(logits, torch.from_numpy(data["val_y"]).to(device)).item())
            val_acc = float((logits.argmax(1).cpu().numpy() == data["val_y"]).mean())
        history = [{"epoch": 0, "train_loss": None, "train_acc": None, "val_loss": val_loss, "val_acc": val_acc}]
        print(f"skip_training val_acc={val_acc:.3f}", flush=True)
    else:
        history = train_linear_head(
            model,
            train_features,
            data["train_y"],
            val_features,
            data["val_y"],
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    out = Path(args.out or f"reports/{args.dataset}_{args.fm}_fm_feature_audit_{args.label_mode}.json")
    checkpoint = Path(args.checkpoint or f"reports/{args.dataset}_{args.fm}_fm_feature_probe_{args.label_mode}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": model.head.state_dict(), "history": history, "args": vars(args)}, checkpoint)

    layer_names = ("feature_identity", "logits")
    capture = LayerCapture(model.eval(), layer_names)
    try:
        pairwise = layer_pairwise_isometry(
            audit_x,
            capture,
            layer_names=layer_names,
            device=device,
            dt=1.0 / spec["sfreq"],
            n_pairs=args.n_pairs,
            pair_batch_size=args.pair_batch_size,
            seed=args.seed,
        )
        band = layer_band_orthogonality(
            audit_x,
            capture,
            layer_names=layer_names,
            device=device,
            sfreq=spec["sfreq"],
            n_epochs=args.band_epochs,
            seed=args.seed,
        )
        local = layer_local_directional_distortion(
            audit_x,
            capture,
            layer_names=layer_names,
            device=device,
            dt=1.0 / spec["sfreq"],
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
        "dataset": data["dataset_meta"],
        "fm_preprocessing": {
            "fm_id": args.fm,
            "target_sfreq": spec["sfreq"],
            "target_n_times": spec["n_times"],
            "adapted_ch_names": adapted_ch_names,
            "train_fm_shape": list(train_x.shape),
            "val_fm_shape": list(val_x.shape),
            "audit_fm_shape": list(audit_x.shape),
            "audit_indices": audit_idx.tolist(),
        },
        "model": {
            "arch": args.fm,
            "source": spec["source"],
            "repo": spec["repo"],
            "feature_dim": spec["feature_dim"],
            "checkpoint": str(checkpoint),
            "n_parameters_fm": int(sum(p.numel() for p in model.fm.parameters())),
            "n_parameters_head": int(sum(p.numel() for p in model.head.parameters())),
            "layer_names": layer_names,
        },
        "training": {
            "history": history,
            "best_val_acc": float(max(row["val_acc"] for row in history)),
            "final_val_acc": float(history[-1]["val_acc"]),
            "final_train_acc": None if history[-1]["train_acc"] is None else float(history[-1]["train_acc"]),
            "label_mode": args.label_mode,
            "skip_training": bool(args.skip_training),
            "probe": "frozen_fm_linear_head",
        },
        "layers": layers,
    }
    write_json(out, payload)
    print("wrote", out, flush=True)
    print("best_val_acc", f"{payload['training']['best_val_acc']:.3f}", flush=True)
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
