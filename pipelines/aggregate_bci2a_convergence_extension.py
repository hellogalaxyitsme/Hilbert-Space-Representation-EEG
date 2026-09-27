#!/usr/bin/env python3
"""Aggregate BCI IV 2a convergence extension runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/tier2_convergence")
    parser.add_argument("--out-dir", default="reports/tier2_convergence_stats")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.reports_dir).glob("*_bci2a_convergence_true_seed*.json")):
        payload = read_json(path)
        rows.append(
            {
                "dataset": "bci2a",
                "protocol": payload["dataset"]["protocol"],
                "arch": payload["model"]["arch"],
                "seed": payload["run"]["seed"],
                "max_epochs": payload["run"]["max_epochs"],
                "patience": payload["run"]["patience"],
                "stopped_epoch": payload["summary"]["stopped_epoch"],
                "best_epoch": payload["summary"]["best_epoch"],
                "best_val_acc": payload["summary"]["best_val_acc"],
                "n_parameters": payload["model"]["n_parameters"],
                "source": str(path),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arch"]].append(row)
    summary = []
    for arch, vals in sorted(grouped.items()):
        acc = np.asarray([float(v["best_val_acc"]) for v in vals], dtype=float)
        summary.append(
            {
                "dataset": "bci2a",
                "protocol": vals[0]["protocol"],
                "arch": arch,
                "n_seeds": len(vals),
                "best_val_acc_mean": float(acc.mean()),
                "best_val_acc_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "best_val_acc_min": float(acc.min()),
                "best_val_acc_max": float(acc.max()),
                "mean_best_epoch": float(np.mean([float(v["best_epoch"]) for v in vals])),
                "mean_stopped_epoch": float(np.mean([float(v["stopped_epoch"]) for v in vals])),
            }
        )
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "bci2a_convergence_extension_rows.csv", rows)
    write_csv(out_dir / "bci2a_convergence_extension_summary.csv", summary)
    lines = [
        "# BCI IV 2a EEGConformer/TSception Convergence Extension",
        "",
        "Protocol: cross-subject A01T-A08T training with A09T validation/test proxy, using the same preconverted BCI IV 2a FIF cache as the main HSDD experiments.",
        "",
        "| Architecture | Seeds | Best val acc mean +/- std | Range | Mean best epoch |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['arch']}` | {row['n_seeds']} | {fmt(row['best_val_acc_mean'])} +/- {fmt(row['best_val_acc_std'])} | "
            f"[{fmt(row['best_val_acc_min'])}, {fmt(row['best_val_acc_max'])}] | {fmt(row['mean_best_epoch'])} |"
        )
    lines += [
        "",
        "Literature-comparability note: this is a strict cross-subject split, not within-subject/session-wise evaluation; manuscript comparisons must cite cross-subject BCI IV 2a numbers rather than within-subject leaderboard values.",
    ]
    (out_dir / "bci2a_convergence_extension_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
