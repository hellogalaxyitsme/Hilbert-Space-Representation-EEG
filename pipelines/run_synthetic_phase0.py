#!/usr/bin/env python3
"""Run Phase 0 synthetic feasibility diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsdd.diagnostics import (
    band_orthogonality_report,
    local_directional_distortion,
    pairwise_isometry_report,
)
from hsdd.feature_maps import BandMixingMap, IdentityMap, LinearProjection, ReluProjection
from hsdd.io import write_json
from hsdd.synthetic import fft_band_components, make_sine_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/synthetic_phase0.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-epochs", type=int, default=96)
    parser.add_argument("--n-channels", type=int, default=8)
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--duration", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, labels, dt = make_sine_epochs(
        n_epochs=args.n_epochs,
        n_channels=args.n_channels,
        sfreq=args.sfreq,
        duration=args.duration,
        seed=args.seed,
    )
    input_dim = x.shape[1] * x.shape[2]
    maps = {
        "identity": IdentityMap(),
        "orthogonal_projection_64d": LinearProjection(input_dim, 64, seed=args.seed),
        "relu_projection_64d": ReluProjection(input_dim, 64, seed=args.seed),
        "intentional_band_mixer": BandMixingMap(),
    }
    band_components, band_names = fft_band_components(x[0], sfreq=args.sfreq)

    results = {
        "dataset": {
            "kind": "synthetic_sine_epochs",
            "shape": list(x.shape),
            "label_counts": {
                str(label): int((labels == label).sum()) for label in sorted(set(labels))
            },
            "dt": dt,
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
                n_points=12,
                n_directions=6,
                seed=args.seed,
            ),
        }

    out = Path(args.out)
    write_json(out, results)
    print(f"wrote {out}")
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
            f"{odi.input_mean_abs_offdiag_cosine:.4e}",
            "output_odi=",
            f"{odi.output_mean_abs_offdiag_cosine:.4f}",
        )


if __name__ == "__main__":
    main()
