"""SEED-IV raw EEG cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SEED_IV_LABEL_NAMES = ["neutral", "sad", "fear", "happy"]
SEED_IV_SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

SEED_IV_CH_NAMES = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POz",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "Oz",
    "O2",
    "CB2",
]


def parse_subject_id(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def parse_val_subjects(text: str) -> set[int]:
    return {int(item) for item in text.replace(",", " ").split() if item.strip()}


def balanced_indices(y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(y.astype(int).tolist())):
        idx = np.flatnonzero(y == label)
        if max_per_class > 0 and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def label_counts(y: np.ndarray, label_names: list[str] = SEED_IV_LABEL_NAMES) -> dict[str, int]:
    return {label_names[i]: int((y == i).sum()) for i in range(len(label_names))}


def load_seediv_cache(
    path: Path,
    *,
    val_subjects: str,
    seed: int,
    max_train_per_class: int,
    max_val_per_class: int,
) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subject = data["subject"].astype(np.int64)
    session = data["session"].astype(np.int64)
    trial = data["trial"].astype(np.int64)
    window = data["window"].astype(np.int64)
    sfreq = float(data["sfreq"][0])
    label_names = [str(v) for v in data["label_names"]]
    channel_names = [str(v) for v in data["channel_names"]]
    val_ids = parse_val_subjects(val_subjects)
    val_mask = np.asarray([int(s) in val_ids for s in subject])
    train_mask = ~val_mask
    train_idx_all = np.flatnonzero(train_mask)
    val_idx_all = np.flatnonzero(val_mask)
    train_sub = balanced_indices(y[train_idx_all], max_train_per_class, seed)
    val_sub = balanced_indices(y[val_idx_all], max_val_per_class, seed)
    train_idx = train_idx_all[train_sub]
    val_idx = val_idx_all[val_sub]
    return {
        "x": x,
        "y": y,
        "subject": subject,
        "session": session,
        "trial": trial,
        "window": window,
        "sfreq": sfreq,
        "dt": 1.0 / sfreq,
        "label_names": label_names,
        "channel_names": channel_names,
        "val_subjects": sorted(val_ids),
        "train_subjects": sorted(set(subject.astype(int).tolist()) - val_ids),
        "train_x": x[train_idx],
        "train_y": y[train_idx],
        "val_x": x[val_idx],
        "val_y": y[val_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "cache_meta": {
            key: data[key].tolist() if getattr(data[key], "shape", ()) != () else data[key].item()
            for key in data.files
            if key.endswith("_meta")
        },
    }
