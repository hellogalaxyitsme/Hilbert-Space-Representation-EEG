#!/usr/bin/env python3
"""Aggregate raw ODI minus the analytic Gaussian reference from compact rows.

The closed-form reference is alpha_d = E|<u,v>| for independent random unit
vectors in output dimension d. Historical composite-null fields, if present,
are copied only as provenance and are never reinterpreted as alpha_d.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


REQUIRED = {"dataset", "arch", "mode", "seed", "epoch", "layer", "output_dim", "actual_raw_odi", "analytic_gaussian_alpha_d"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def analytic_gaussian_alpha(dimension: int) -> float:
    """Return E|<u,v>| for independent unit vectors in R^dimension."""
    if dimension < 1:
        raise ValueError("output dimension must be positive")
    return math.exp(
        math.lgamma(dimension / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension + 1) / 2.0)
    )


def read_compact(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError(f"{path} is empty or lacks analytic-reference columns")
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in ("dataset", "arch", "mode", "seed", "epoch", "layer"))
        if key in seen:
            raise ValueError(f"duplicate analytic record: {key}")
        seen.add(key)
        dimension = int(row["output_dim"])
        actual = float(row["actual_raw_odi"])
        supplied_alpha = float(row["analytic_gaussian_alpha_d"])
        if dimension < 1:
            raise ValueError(f"invalid output dimension for {key}")
        if not math.isfinite(actual) or not 0.0 <= actual <= 1.0:
            raise ValueError(f"raw ODI outside [0, 1] for {key}")
        expected_alpha = analytic_gaussian_alpha(dimension)
        if not math.isfinite(supplied_alpha) or abs(supplied_alpha - expected_alpha) > 1e-10:
            raise ValueError(f"analytic alpha does not match output dimension for {key}")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["arch"]].append(row)
    result = []
    for arch, group in sorted(grouped.items()):
        actual = [float(row["actual_raw_odi"]) for row in group]
        # Preserve published compact-table rounding exactly after validating
        # that each stored alpha is the independently computed alpha_d.
        alpha = [float(row["analytic_gaussian_alpha_d"]) for row in group]
        excess = [left - right for left, right in zip(actual, alpha)]
        result.append({
            "arch": arch, "n_rows": len(group), "n_datasets": len({row["dataset"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}), "n_modes": len({row["mode"] for row in group}),
            "n_checkpoints": len({row["epoch"] for row in group}), "actual_raw_odi_mean": mean(actual),
            "analytic_gaussian_alpha_d_mean": mean(alpha), "analytic_gaussian_excess_mean": mean(excess),
            "positive_analytic_excess_fraction": mean([value > 0 for value in excess]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Compact analytic_gaussian_rows.csv")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = read_compact(args.input)
    summary = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"canonical_rows={len(rows)}; architectures={len(summary)}; wrote={args.summary}")


if __name__ == "__main__":
    main()
