#!/usr/bin/env python3
"""Prepare a raw-window SEED-IV cache for HSDD experiments."""

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
from scipy.io import loadmat, whosmat

from seediv_utils import SEED_IV_CH_NAMES, SEED_IV_LABEL_NAMES, SEED_IV_SESSION_LABELS, label_counts, parse_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument(
        "--max-windows-per-trial",
        type=int,
        default=8,
        help="Use 0 for every non-overlapping window. Default is a bounded first-pass raw cache.",
    )
    return parser.parse_args()


def select_window_starts(n_times: int, window: int, stride: int, max_windows: int) -> np.ndarray:
    starts = np.arange(0, n_times - window + 1, stride, dtype=int)
    if max_windows > 0 and len(starts) > max_windows:
        keep = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = starts[keep]
    return starts


def trial_key(path: Path, trial: int) -> str:
    suffix = f"_eeg{trial}"
    keys = [name for name, _shape, _dtype in whosmat(path) if name.endswith(suffix)]
    if len(keys) != 1:
        raise KeyError(f"expected one key ending {suffix} in {path}, found {keys}")
    return keys[0]


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    epoch = epoch.astype(np.float32, copy=False)
    epoch = epoch - epoch.mean(axis=-1, keepdims=True)
    scale = float(epoch.std())
    return epoch / max(scale, 1e-6)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    raw_root = root / "eeg_raw_data"
    out = Path(args.out)
    window = int(round(args.window_sec * args.sfreq))
    stride = int(round(args.stride_sec * args.sfreq))
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    sessions: list[int] = []
    trials: list[int] = []
    windows: list[int] = []

    for session_id in sorted(SEED_IV_SESSION_LABELS):
        session_dir = raw_root / str(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(session_dir)
        labels = SEED_IV_SESSION_LABELS[session_id]
        for mat_path in sorted(session_dir.glob("*.mat"), key=lambda p: parse_subject_id(p)):
            subject_id = parse_subject_id(mat_path)
            mat = None
            n_added = 0
            for trial_id, label in enumerate(labels, start=1):
                key = trial_key(mat_path, trial_id)
                if mat is None:
                    mat = loadmat(mat_path)
                raw = np.asarray(mat[key], dtype=np.float32)
                if raw.shape[0] != len(SEED_IV_CH_NAMES):
                    raise ValueError(f"unexpected channel count in {mat_path}:{key}: {raw.shape}")
                starts = select_window_starts(raw.shape[-1], window, stride, args.max_windows_per_trial)
                for win_idx, start in enumerate(starts):
                    xs.append(normalize_epoch(raw[:, start : start + window]))
                    ys.append(int(label))
                    subjects.append(subject_id)
                    sessions.append(session_id)
                    trials.append(trial_id)
                    windows.append(win_idx)
                    n_added += 1
            print(f"loaded session={session_id} subject={subject_id:02d} windows={n_added}", flush=True)

    x = np.stack(xs).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=x,
        y=y,
        subject=np.asarray(subjects, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
        window=np.asarray(windows, dtype=np.int64),
        sfreq=np.asarray([args.sfreq], dtype=np.float32),
        channel_names=np.asarray(SEED_IV_CH_NAMES),
        label_names=np.asarray(SEED_IV_LABEL_NAMES),
        root_meta=np.asarray(str(root)),
        window_sec_meta=np.asarray(args.window_sec),
        stride_sec_meta=np.asarray(args.stride_sec),
        max_windows_per_trial_meta=np.asarray(args.max_windows_per_trial),
    )
    print("wrote", out, flush=True)
    print("x_shape", list(x.shape), "label_counts", label_counts(y), flush=True)


if __name__ == "__main__":
    main()
