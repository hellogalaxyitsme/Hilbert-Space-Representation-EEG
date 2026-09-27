#!/usr/bin/env python3
"""Run Phase 0 diagnostics on one preconverted BCI IV 2a subject/session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from hsdd.diagnostics import (
    band_orthogonality_report,
    local_directional_distortion,
    pairwise_isometry_report,
)
from hsdd.feature_maps import BandMixingMap, IdentityMap, LinearProjection, ReluProjection
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-dir",
        default=None,
    )
    parser.add_argument("--out", default="reports/bci2a_a01t_phase0.json")
    parser.add_argument("--max-epochs", type=int, default=96)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_bci2a_epochs(
    session_dir: Path,
    *,
    tmin: float,
    tmax: float,
    max_epochs: int,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    import mne

    raw_path = session_dir / f"{session_dir.name}_raw_cleaned.fif"
    if not raw_path.exists():
        raw_path = session_dir / f"{session_dir.name}_raw.fif"
    events_path = session_dir / f"{session_dir.name}_events.npy"
    if not raw_path.exists() or not events_path.exists():
        raise FileNotFoundError(f"missing raw/events files under {session_dir}")

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
    events = np.load(events_path)
    event_id = {
        "left_hand": 769,
        "right_hand": 770,
        "feet": 771,
        "tongue": 772,
    }
    keep_codes = set(event_id.values())
    events = events[np.asarray([event[2] in keep_codes for event in events])]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise RuntimeError("no BCI IV 2a cue epochs survived epoching")

    x = epochs.get_data(copy=True)[:max_epochs]
    labels = epochs.events[: len(x), 2]
    channel_names = [raw.ch_names[idx] for idx in picks]

    x = x.astype(np.float64)
    x -= x.mean(axis=-1, keepdims=True)
    scale = np.std(x, axis=(-2, -1), keepdims=True)
    x = x / np.maximum(scale, 1e-12)
    return x, labels.astype(int), 1.0 / float(raw.info["sfreq"]), channel_names


def main() -> None:
    args = parse_args()
    session_dir = Path(args.session_dir)
    x, labels, dt, channel_names = load_bci2a_epochs(
        session_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        max_epochs=args.max_epochs,
    )
    input_dim = x.shape[1] * x.shape[2]
    maps = {
        "identity": IdentityMap(),
        "orthogonal_projection_128d": LinearProjection(input_dim, 128, seed=args.seed),
        "relu_projection_128d": ReluProjection(input_dim, 128, seed=args.seed),
        "intentional_band_mixer": BandMixingMap(),
    }
    sfreq = 1.0 / dt
    band_components, band_names = fft_band_components(x[0], sfreq=sfreq)

    results = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif",
            "session_dir": str(session_dir),
            "shape": list(x.shape),
            "dt": dt,
            "sfreq": sfreq,
            "tmin": args.tmin,
            "tmax": args.tmax,
            "channel_names": channel_names,
            "label_counts": {
                str(label): int((labels == label).sum()) for label in sorted(set(labels))
            },
        },
        "maps": {},
    }

    for name, fmap in maps.items():
        results["maps"][name] = {
            "pairwise_isometry": pairwise_isometry_report(
                x,
                fmap,
                dt=dt,
                n_pairs=2048,
                seed=args.seed,
            ),
            "band_orthogonality": band_orthogonality_report(
                band_components,
                fmap,
                band_names=band_names,
            ),
            "local_directional_distortion": local_directional_distortion(
                x,
                fmap,
                dt=dt,
                n_points=8,
                n_directions=4,
                seed=args.seed,
            ),
        }

    write_json(args.out, results)
    print(f"wrote {args.out}")
    print("dataset_shape", x.shape, "labels", results["dataset"]["label_counts"])
    for name, report in results["maps"].items():
        iso = report["pairwise_isometry"]
        odi = report["band_orthogonality"]
        print(
            name,
            "scaled_iso_mean=",
            f"{iso.scaled_abs_ratio_error_mean:.4f}",
            "spearman=",
            f"{iso.distance_spearman:.4f}",
            "input_odi=",
            f"{odi.input_mean_abs_offdiag_cosine:.4f}",
            "output_odi=",
            f"{odi.output_mean_abs_offdiag_cosine:.4f}",
        )


if __name__ == "__main__":
    main()
