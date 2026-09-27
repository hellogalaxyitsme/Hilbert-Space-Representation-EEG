"""BNCI2014-009 P300 raw EEG cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


P300_LABEL_NAMES = ["nontarget", "target"]
P300_CH_NAMES = [
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "P3",
    "P4",
    "PO7",
    "PO8",
    "F3",
    "F4",
    "FCz",
    "C3",
    "C4",
    "CP3",
    "CPz",
    "CP4",
]


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


def label_counts(y: np.ndarray, label_names: list[str] = P300_LABEL_NAMES) -> dict[str, int]:
    return {label_names[i]: int((y == i).sum()) for i in range(len(label_names))}


def load_p300_cache(
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
    run = data["run"].astype(np.int64)
    event = data["event"].astype(np.int64)
    onset = data["onset"].astype(np.int64)
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
        "run": run,
        "event": event,
        "onset": onset,
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
