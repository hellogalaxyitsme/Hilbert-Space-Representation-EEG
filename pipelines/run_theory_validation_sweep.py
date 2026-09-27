#!/usr/bin/env python3
"""Validate closed-form HSDD theory constants by simulation.

This checks two reported-scope constants:

1. alpha_d = E |<g,h>|/(||g||||h||) for independent Gaussian vectors in R^d.
2. 1/pi uncentered ReLU coherence for uncorrelated Gaussian preactivations.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/theory_validation")
    parser.add_argument("--dims", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--relu-n", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def alpha_d(dim: int) -> float:
    if dim <= 1:
        return 1.0
    return float(math.exp(gammaln(dim / 2.0) - 0.5 * math.log(math.pi) - gammaln((dim + 1) / 2.0)))


def simulate_alpha(dim: int, n_pairs: int, rng: np.random.Generator) -> tuple[float, float]:
    g = rng.normal(size=(n_pairs, dim))
    h = rng.normal(size=(n_pairs, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    vals = np.abs(np.sum(g * h, axis=1))
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))


def relu_validation(n: int, rng: np.random.Generator) -> dict[str, float]:
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    rx = np.maximum(x, 0.0)
    ry = np.maximum(y, 0.0)
    uncentered = float(np.mean(rx * ry) / math.sqrt(np.mean(rx * rx) * np.mean(ry * ry)))
    centered = float(np.corrcoef(rx, ry)[0, 1])
    return {
        "n": n,
        "measured_uncentered_relu_coherence": uncentered,
        "theory_uncentered_relu_coherence": 1.0 / math.pi,
        "abs_error_uncentered": abs(uncentered - 1.0 / math.pi),
        "measured_centered_relu_correlation": centered,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dims = [int(v) for v in args.dims.replace(",", " ").split()]

    alpha_rows = []
    for dim in dims:
        measured, se = simulate_alpha(dim, args.n_pairs, rng)
        theory = alpha_d(dim)
        alpha_rows.append(
            {
                "dim": dim,
                "n_pairs": args.n_pairs,
                "measured_alpha_d": measured,
                "mc_standard_error": se,
                "theory_alpha_d": theory,
                "abs_error": abs(measured - theory),
                "z_error": (measured - theory) / max(se, 1e-12),
            }
        )
    relu_row = relu_validation(args.relu_n, rng)

    write_csv(out_dir / "alpha_d_validation.csv", alpha_rows)
    write_csv(out_dir / "relu_one_over_pi_validation.csv", [relu_row])

    lines = [
        "# Theory-vs-Measurement Validation Sweep",
        "",
        "This synthetic sweep validates the two closed-form constants used by the HSDD theorem backbone.",
        "",
        "## Random-Projection alpha_d",
        "",
        "| d | measured | theory | abs error | MC SE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in alpha_rows:
        lines.append(
            f"| {row['dim']} | {row['measured_alpha_d']:.6f} | {row['theory_alpha_d']:.6f} | "
            f"{row['abs_error']:.6f} | {row['mc_standard_error']:.6f} |"
        )
    lines += [
        "",
        "## ReLU 1/pi Coherence",
        "",
        f"- Measured uncentered ReLU coherence: {relu_row['measured_uncentered_relu_coherence']:.6f}",
        f"- Theory 1/pi: {relu_row['theory_uncentered_relu_coherence']:.6f}",
        f"- Absolute error: {relu_row['abs_error_uncentered']:.6f}",
        f"- Centered post-ReLU correlation check: {relu_row['measured_centered_relu_correlation']:.6f}",
    ]
    (out_dir / "theory_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
