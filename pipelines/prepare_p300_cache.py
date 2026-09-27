#!/usr/bin/env python3
"""Prepare a raw-event BNCI2014-009 P300 cache for HSDD experiments."""

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
from scipy.io import loadmat
from scipy.signal import resample_poly

from p300_utils import P300_CH_NAMES, P300_LABEL_NAMES, label_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--source-sfreq", type=float, default=256.0)
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    return parser.parse_args()


def unwrap(x):
    while isinstance(x, np.ndarray) and x.shape == (1, 1) and x.dtype == object:
        x = x[0, 0]
    return x


def cell_string(value) -> str:
    value = unwrap(value)
    if isinstance(value, np.ndarray):
        return str(value.reshape(-1)[0])
    return str(value)


def channel_names(run_obj) -> list[str]:
    raw = np.asarray(run_obj.channels, dtype=object).reshape(-1)
    return [cell_string(item) for item in raw]


def event_starts(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1)
    nonzero = labels > 0
    prev = np.concatenate([[False], nonzero[:-1]])
    return np.flatnonzero(nonzero & ~prev)


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    epoch = epoch.astype(np.float32, copy=False)
    epoch = epoch - epoch.mean(axis=-1, keepdims=True)
    scale = float(epoch.std())
    return epoch / max(scale, 1e-6)


def maybe_resample(epoch: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return epoch.astype(np.float32, copy=False)
    if np.isclose(source_sfreq, 256.0) and np.isclose(target_sfreq, 200.0):
        return resample_poly(epoch, up=25, down=32, axis=-1).astype(np.float32, copy=False)
    n_times = int(round(epoch.shape[-1] * target_sfreq / source_sfreq))
    from scipy.signal import resample

    return resample(epoch, n_times, axis=-1).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    src_start = int(round(args.tmin * args.source_sfreq))
    src_stop = int(round(args.tmax * args.source_sfreq))
    if src_stop <= src_start:
        raise ValueError("tmax must be greater than tmin")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    runs: list[int] = []
    events: list[int] = []
    onsets: list[int] = []
    stim_groups: list[int] = []
    ch_names_seen: list[str] | None = None

    for subject in range(1, 11):
        mat_path = root / f"A{subject:02d}S.mat"
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        data_cell = mat["data"]
        subject_count = 0
        for run_idx in range(data_cell.shape[1]):
            run_obj = unwrap(data_cell[0, run_idx])
            x_cont = np.asarray(run_obj.X, dtype=np.float32)
            y_cont = np.asarray(run_obj.y).reshape(-1)
            y_stim = np.asarray(run_obj.y_stim).reshape(-1)
            fs = float(np.asarray(run_obj.fs).reshape(-1)[0])
            if not np.isclose(fs, args.source_sfreq):
                raise ValueError(f"unexpected fs in {mat_path} run {run_idx}: {fs}")
            ch_names = channel_names(run_obj)
            if ch_names_seen is None:
                ch_names_seen = ch_names
            elif ch_names != ch_names_seen:
                raise ValueError(f"channel mismatch in {mat_path} run {run_idx}")
            if ch_names != P300_CH_NAMES:
                raise ValueError(f"unexpected channel names in {mat_path}: {ch_names}")

            starts = event_starts(y_cont)
            for event_idx, onset in enumerate(starts):
                label_raw = int(y_cont[onset])
                if label_raw not in (1, 2):
                    continue
                start = int(onset) + src_start
                stop = int(onset) + src_stop
                if start < 0 or stop > x_cont.shape[0]:
                    continue
                epoch = x_cont[start:stop].T
                epoch = maybe_resample(epoch, args.source_sfreq, args.target_sfreq)
                xs.append(normalize_epoch(epoch))
                ys.append(label_raw - 1)
                subjects.append(subject)
                runs.append(run_idx + 1)
                events.append(event_idx)
                onsets.append(int(onset))
                stim_groups.append(int(y_stim[onset]))
                subject_count += 1
        print(f"loaded subject={subject:02d} events={subject_count}", flush=True)

    x = np.stack(xs).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=x,
        y=y,
        subject=np.asarray(subjects, dtype=np.int64),
        run=np.asarray(runs, dtype=np.int64),
        event=np.asarray(events, dtype=np.int64),
        onset=np.asarray(onsets, dtype=np.int64),
        stim_group=np.asarray(stim_groups, dtype=np.int64),
        sfreq=np.asarray([args.target_sfreq], dtype=np.float32),
        source_sfreq=np.asarray([args.source_sfreq], dtype=np.float32),
        channel_names=np.asarray(P300_CH_NAMES),
        label_names=np.asarray(P300_LABEL_NAMES),
        root_meta=np.asarray(str(root)),
        tmin_meta=np.asarray(args.tmin),
        tmax_meta=np.asarray(args.tmax),
        extraction_meta=np.asarray("rising_edge_of_raw_y_post_stimulus"),
    )
    print("wrote", out, flush=True)
    print("x_shape", list(x.shape), "label_counts", label_counts(y), flush=True)


if __name__ == "__main__":
    main()
