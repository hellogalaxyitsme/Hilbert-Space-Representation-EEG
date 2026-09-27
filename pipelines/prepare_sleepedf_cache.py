#!/usr/bin/env python3
"""Prepare a compact Sleep-EDF cache for HSDD architecture audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
LABEL_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3_N4",
    4: "REM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default="prepared/sleepedf_sc20_fpzcz_pzoz.npz")
    parser.add_argument("--max-recordings", type=int, default=20)
    parser.add_argument("--wake-context-epochs", type=int, default=60)
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument("--channels", default="EEG Fpz-Cz,EEG Pz-Oz")
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def pair_files(root: Path) -> list[tuple[Path, Path]]:
    psg_files = sorted(root.glob("*-PSG.edf"))
    hyp_files = sorted(root.glob("*Hypnogram.edf"))
    hyp_by_record = {path.name[:6]: path for path in hyp_files}
    pairs = []
    for psg in psg_files:
        key = psg.name[:6]
        if key in hyp_by_record:
            pairs.append((psg, hyp_by_record[key]))
    return pairs


def annotation_epoch_labels(annotations, n_epochs: int, epoch_seconds: float) -> np.ndarray:
    labels = np.full(n_epochs, -1, dtype=np.int64)
    for onset, duration, desc in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        desc = str(desc)
        if desc not in LABEL_MAP:
            continue
        start = int(np.floor(float(onset) / epoch_seconds + 1e-6))
        stop = int(np.ceil((float(onset) + float(duration)) / epoch_seconds - 1e-6))
        start = max(start, 0)
        stop = min(stop, n_epochs)
        labels[start:stop] = LABEL_MAP[desc]
    return labels


def load_recording(
    psg_path: Path,
    hyp_path: Path,
    *,
    channel_names: list[str],
    epoch_seconds: float,
    wake_context_epochs: int,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    raw.pick(channel_names)
    sfreq = float(raw.info["sfreq"])
    n_times = int(round(epoch_seconds * sfreq))
    n_epochs = raw.n_times // n_times
    annotations = mne.read_annotations(hyp_path)
    labels = annotation_epoch_labels(annotations, n_epochs, epoch_seconds)
    valid = labels >= 0
    sleep = valid & (labels != 0)
    if not np.any(sleep):
        raise RuntimeError(f"no sleep stages found in {psg_path.name}")
    sleep_idx = np.flatnonzero(sleep)
    lo = max(0, int(sleep_idx[0]) - wake_context_epochs)
    hi = min(n_epochs, int(sleep_idx[-1]) + wake_context_epochs + 1)
    keep = np.flatnonzero(valid & (np.arange(n_epochs) >= lo) & (np.arange(n_epochs) < hi))

    epochs = []
    kept_labels = []
    for idx in keep:
        start = int(idx * n_times)
        stop = start + n_times
        x = raw.get_data(start=start, stop=stop).astype(np.float32)
        if x.shape[-1] != n_times:
            continue
        x -= x.mean(axis=-1, keepdims=True)
        scale = x.std(axis=(-2, -1), keepdims=True)
        x = x / np.maximum(scale, 1e-6)
        epochs.append(x)
        kept_labels.append(int(labels[idx]))
    return np.asarray(epochs, dtype=np.float32), np.asarray(kept_labels, dtype=np.int64), raw.ch_names, sfreq


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    channel_names = [item.strip() for item in args.channels.split(",") if item.strip()]
    pairs = pair_files(root)
    if args.max_recordings > 0:
        pairs = pairs[: args.max_recordings]
    if not pairs:
        raise FileNotFoundError(f"no Sleep-EDF pairs found under {root}")

    xs = []
    ys = []
    recording_ids = []
    recording_names = []
    sfreq = None
    used_channels = None
    for ridx, (psg, hyp) in enumerate(pairs):
        x, y, used_channels, sfreq = load_recording(
            psg,
            hyp,
            channel_names=channel_names,
            epoch_seconds=args.epoch_seconds,
            wake_context_epochs=args.wake_context_epochs,
        )
        xs.append(x)
        ys.append(y)
        recording_ids.append(np.full(len(y), ridx, dtype=np.int64))
        recording_names.append(psg.name)
        counts = {LABEL_NAMES[k]: int((y == k).sum()) for k in sorted(LABEL_NAMES)}
        print("loaded", psg.name, x.shape, counts, flush=True)

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    recording_ids_all = np.concatenate(recording_ids, axis=0)
    np.savez_compressed(
        out,
        x=x_all,
        y=y_all,
        recording_id=recording_ids_all,
        recording_names=np.asarray(recording_names),
        channel_names=np.asarray(used_channels),
        sfreq=np.asarray([sfreq], dtype=np.float32),
        label_names=np.asarray([LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]),
        epoch_seconds=np.asarray([args.epoch_seconds], dtype=np.float32),
    )
    meta = {
        "root": str(root),
        "out": str(out),
        "n_recordings": len(recording_names),
        "recording_names": recording_names,
        "shape": list(x_all.shape),
        "sfreq": sfreq,
        "channel_names": used_channels,
        "label_counts": {LABEL_NAMES[k]: int((y_all == k).sum()) for k in sorted(LABEL_NAMES)},
        "wake_context_epochs": args.wake_context_epochs,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("wrote", out, meta["shape"], meta["label_counts"], flush=True)


if __name__ == "__main__":
    main()
