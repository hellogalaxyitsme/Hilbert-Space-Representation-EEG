#!/usr/bin/env python3
"""Reproduce the four supplied regularizer confidence intervals from paired runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.aggregate_tier3_bci2a_regularizer import bootstrap_ci


ENDPOINTS = {
    "band_noise_mean_accuracy": "delta_band_noise_acc_arm4_minus_arm2",
    "channel_dropout_mean_accuracy": "delta_channel_dropout_acc_arm4_minus_arm2",
    "clean_ECE": "delta_clean_ece_arm4_minus_arm2",
    "clean_accuracy": "delta_clean_acc_arm4_minus_arm2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45:
        raise ValueError(f"expected 45 paired runs, found {len(rows)}")
    output = []
    for endpoint, field in ENDPOINTS.items():
        values = [float(row[field]) for row in rows]
        low, high = bootstrap_ci(values, n_boot=5000, seed=1729)
        output.append({"endpoint": endpoint, "estimate": sum(values) / len(values), "ci_low": low, "ci_high": high, "n_pairs": len(values)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"rows": len(output), "n_pairs": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
