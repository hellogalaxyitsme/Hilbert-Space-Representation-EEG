#!/usr/bin/env python3
"""Run Phase 0 diagnostics across the available BCI Competition IV 2a dataset."""

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
from hsdd.io import to_jsonable, write_json
from hsdd.synthetic import fft_band_components


EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "feet": 771,
    "tongue": 772,
}
UNKNOWN_EVAL_EVENT_ID = {"unknown_eval_cue": 783}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
    )
    parser.add_argument("--out", default="reports/bci2a_full_phase0.json")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--max-epochs-per-session", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=16384)
    parser.add_argument("--band-epochs-per-session", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


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
    keep_codes = set(EVENT_ID.values())
    events_labeled = events[np.asarray([event[2] in keep_codes for event in events])]
    event_id = EVENT_ID
    if len(events_labeled) == 0:
        unknown_codes = set(UNKNOWN_EVAL_EVENT_ID.values())
        events_labeled = events[np.asarray([event[2] in unknown_codes for event in events])]
        event_id = UNKNOWN_EVAL_EVENT_ID
    events = events_labeled
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
        raise RuntimeError(f"no cue epochs survived epoching for {session_dir.name}")

    x = epochs.get_data(copy=True)
    labels = epochs.events[:, 2]
    if max_epochs > 0:
        x = x[:max_epochs]
        labels = labels[:max_epochs]

    x = x.astype(np.float64)
    x -= x.mean(axis=-1, keepdims=True)
    scale = np.std(x, axis=(-2, -1), keepdims=True)
    x = x / np.maximum(scale, 1e-12)
    channel_names = [raw.ch_names[idx] for idx in picks]
    return x, labels.astype(int), 1.0 / float(raw.info["sfreq"]), channel_names


def band_summary(
    x: np.ndarray,
    fmap,
    *,
    sfreq: float,
    n_epochs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = min(n_epochs, len(x))
    indices = rng.choice(len(x), size=n, replace=False)
    input_odi = []
    output_odi = []
    collapse = []
    input_mats = []
    output_mats = []
    band_names = None
    for idx in indices:
        components, names = fft_band_components(x[idx], sfreq=sfreq)
        report = band_orthogonality_report(components, fmap, band_names=names)
        band_names = report.band_names
        input_odi.append(report.input_mean_abs_offdiag_cosine)
        output_odi.append(report.output_mean_abs_offdiag_cosine)
        collapse.append(report.collapse_factor)
        input_mats.append(report.input_cosine_matrix)
        output_mats.append(report.output_cosine_matrix)

    return {
        "band_names": band_names,
        "n_epochs": int(n),
        "input_mean_abs_offdiag_cosine_mean": float(np.mean(input_odi)),
        "input_mean_abs_offdiag_cosine_std": float(np.std(input_odi)),
        "output_mean_abs_offdiag_cosine_mean": float(np.mean(output_odi)),
        "output_mean_abs_offdiag_cosine_std": float(np.std(output_odi)),
        "collapse_factor_mean": float(np.mean(collapse)),
        "collapse_factor_std": float(np.std(collapse)),
        "input_cosine_matrix_mean": np.mean(input_mats, axis=0),
        "output_cosine_matrix_mean": np.mean(output_mats, axis=0),
    }


def summarize_labels(labels: np.ndarray) -> dict[str, int]:
    return {str(label): int((labels == label).sum()) for label in sorted(set(labels))}


def run_maps(
    x: np.ndarray,
    labels: np.ndarray,
    maps: dict,
    *,
    dt: float,
    sfreq: float,
    n_pairs: int,
    band_epochs: int,
    seed: int,
) -> dict:
    results = {}
    for name, fmap in maps.items():
        results[name] = {
            "pairwise_isometry": pairwise_isometry_report(
                x,
                fmap,
                dt=dt,
                n_pairs=n_pairs,
                seed=seed,
            ),
            "band_orthogonality": band_summary(
                x,
                fmap,
                sfreq=sfreq,
                n_epochs=band_epochs,
                seed=seed,
            ),
            "local_directional_distortion": local_directional_distortion(
                x,
                fmap,
                dt=dt,
                n_points=min(16, len(x)),
                n_directions=4,
                seed=seed,
            ),
        }
    return results


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    sessions = discover_sessions(root)
    if not sessions:
        raise FileNotFoundError(f"no AxxT/AxxE session folders found under {root}")

    loaded = []
    for session in sessions:
        try:
            x, labels, dt, channel_names = load_session(
                session,
                tmin=args.tmin,
                tmax=args.tmax,
                max_epochs=args.max_epochs_per_session,
            )
        except Exception as exc:
            print(f"skip {session.name}: {exc}")
            continue
        loaded.append(
            {
                "name": session.name,
                "path": str(session),
                "x": x,
                "labels": labels,
                "dt": dt,
                "sfreq": 1.0 / dt,
                "channel_names": channel_names,
            }
        )
        print("loaded", session.name, x.shape, summarize_labels(labels))

    if not loaded:
        raise RuntimeError("no sessions could be loaded")

    input_dim = loaded[0]["x"].shape[1] * loaded[0]["x"].shape[2]
    maps = {
        "identity": IdentityMap(),
        "orthogonal_projection_128d": LinearProjection(input_dim, 128, seed=args.seed),
        "relu_projection_128d": ReluProjection(input_dim, 128, seed=args.seed),
        "intentional_band_mixer": BandMixingMap(),
    }

    session_reports = []
    for item in loaded:
        report = {
            "name": item["name"],
            "path": item["path"],
            "shape": list(item["x"].shape),
            "dt": item["dt"],
            "sfreq": item["sfreq"],
            "label_counts": summarize_labels(item["labels"]),
            "maps": run_maps(
                item["x"],
                item["labels"],
                maps,
                dt=item["dt"],
                sfreq=item["sfreq"],
                n_pairs=min(args.n_pairs, max(1024, len(item["x"]) * 64)),
                band_epochs=args.band_epochs_per_session,
                seed=args.seed,
            ),
        }
        session_reports.append(report)
        print("diagnosed", item["name"])

    x_all = np.concatenate([item["x"] for item in loaded], axis=0)
    labels_all = np.concatenate([item["labels"] for item in loaded], axis=0)
    pooled = {
        "shape": list(x_all.shape),
        "dt": loaded[0]["dt"],
        "sfreq": loaded[0]["sfreq"],
        "label_counts": summarize_labels(labels_all),
        "maps": run_maps(
            x_all,
            labels_all,
            maps,
            dt=loaded[0]["dt"],
            sfreq=loaded[0]["sfreq"],
            n_pairs=args.n_pairs,
            band_epochs=args.band_epochs_per_session * len(loaded),
            seed=args.seed,
        ),
    }

    payload = {
        "dataset": {
            "kind": "BCI_IV_2a_preconverted_fif_full_available",
            "root": str(root),
            "n_sessions": len(loaded),
            "session_names": [item["name"] for item in loaded],
            "tmin": args.tmin,
            "tmax": args.tmax,
            "max_epochs_per_session": args.max_epochs_per_session,
            "channel_names": loaded[0]["channel_names"],
        },
        "sessions": session_reports,
        "pooled": pooled,
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")
    print("pooled_shape", pooled["shape"], "labels", pooled["label_counts"])
    for name, report in pooled["maps"].items():
        iso = report["pairwise_isometry"]
        band = report["band_orthogonality"]
        print(
            name,
            "scaled_iso_mean=",
            f"{iso.scaled_abs_ratio_error_mean:.4f}",
            "spearman=",
            f"{iso.distance_spearman:.4f}",
            "input_odi=",
            f"{band['input_mean_abs_offdiag_cosine_mean']:.4f}",
            "output_odi=",
            f"{band['output_mean_abs_offdiag_cosine_mean']:.4f}",
        )


if __name__ == "__main__":
    main()
