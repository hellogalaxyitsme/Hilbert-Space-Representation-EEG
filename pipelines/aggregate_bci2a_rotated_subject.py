#!/usr/bin/env python3
"""Aggregate BCI IV 2a rotated held-out-subject robustness reports."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/bci2a_rotated_subject")
    parser.add_argument("--out-dir", default="reports/bci2a_rotated_subject_stats")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


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


def rankdata_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def wilcoxon_exact(x: list[float], y: list[float]) -> dict[str, float | int | str]:
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = len(diffs)
    if n == 0:
        return {"test": "wilcoxon_signed_rank_exact", "n": 0, "statistic": 0.0, "p_value": 1.0}
    ranks = rankdata_abs(np.abs(diffs))
    obs = min(float(ranks[diffs > 0].sum()), float(ranks[diffs < 0].sum()))
    totals = []
    for signs in itertools.product((0, 1), repeat=n):
        w = float(ranks[np.asarray(signs, dtype=bool)].sum())
        totals.append(min(w, float(ranks.sum() - w)))
    p = sum(total <= obs + 1e-12 for total in totals) / len(totals)
    return {"test": "wilcoxon_signed_rank_exact", "n": n, "statistic": obs, "p_value": float(p)}


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return (float(arr[0]) if len(arr) else float("nan"), 0.0)
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.3f}"


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    metric_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json")):
        payload = read_json(path)
        dataset = payload["dataset"]
        arch = payload["model"]["arch"]
        mode = payload["training"]["label_mode"]
        seed = int(payload.get("training", {}).get("seed", 0) or path.stem.rsplit("_seed", 1)[-1])
        heldout = dataset["heldout_subject"]
        metric_rows.append(
            {
                "arch": arch,
                "heldout_subject": heldout,
                "mode": mode,
                "seed": seed,
                "best_val_acc": float(payload["training"]["best_val_acc"]),
                "final_val_acc": float(payload["training"]["final_val_acc"]),
                "final_train_acc": payload["training"]["final_train_acc"],
                "source": str(path),
            }
        )
        for layer, vals in payload.get("layers", {}).items():
            layer_rows.append(
                {
                    "arch": arch,
                    "heldout_subject": heldout,
                    "mode": mode,
                    "seed": seed,
                    "layer": layer,
                    "distance_spearman": vals["pairwise_isometry"]["distance_spearman"],
                    "scaled_ratio_error": vals["pairwise_isometry"]["scaled_abs_ratio_error_mean"],
                    "odi": vals["band_orthogonality"]["output_mean_abs_offdiag_cosine_mean"],
                    "local_stretch": vals["local_directional_distortion"]["stretch_mean"],
                    "source": str(path),
                }
            )

    summary_rows = []
    for key in sorted({(r["arch"], r["mode"]) for r in metric_rows}):
        arch, mode = key
        vals = [float(r["best_val_acc"]) for r in metric_rows if (r["arch"], r["mode"]) == key]
        mean, std = mean_std(vals)
        summary_rows.append(
            {
                "arch": arch,
                "mode": mode,
                "n": len(vals),
                "best_val_acc_mean": mean,
                "best_val_acc_std": std,
                "best_val_acc_pm_std": f"{fmt(mean)} +/- {fmt(std)}",
            }
        )

    subject_rows = []
    for key in sorted({(r["arch"], r["heldout_subject"], r["mode"]) for r in metric_rows}):
        arch, heldout, mode = key
        vals = [
            float(r["best_val_acc"])
            for r in metric_rows
            if (r["arch"], r["heldout_subject"], r["mode"]) == key
        ]
        mean, std = mean_std(vals)
        subject_rows.append(
            {
                "arch": arch,
                "heldout_subject": heldout,
                "mode": mode,
                "n": len(vals),
                "best_val_acc_mean": mean,
                "best_val_acc_std": std,
                "best_val_acc_pm_std": f"{fmt(mean)} +/- {fmt(std)}",
            }
        )

    test_rows = []
    for arch in sorted({r["arch"] for r in metric_rows}):
        pairs = []
        for heldout, seed in sorted({(r["heldout_subject"], r["seed"]) for r in metric_rows if r["arch"] == arch}):
            true = [
                r
                for r in metric_rows
                if r["arch"] == arch and r["heldout_subject"] == heldout and r["seed"] == seed and r["mode"] == "true"
            ]
            shuffled = [
                r
                for r in metric_rows
                if r["arch"] == arch and r["heldout_subject"] == heldout and r["seed"] == seed and r["mode"] == "shuffled"
            ]
            if true and shuffled:
                pairs.append((float(true[0]["best_val_acc"]), float(shuffled[0]["best_val_acc"])))
        true_vals = [p[0] for p in pairs]
        shuffled_vals = [p[1] for p in pairs]
        true_mean, true_std = mean_std(true_vals)
        shuffled_mean, shuffled_std = mean_std(shuffled_vals)
        diffs = np.asarray(true_vals) - np.asarray(shuffled_vals)
        test = wilcoxon_exact(true_vals, shuffled_vals)
        test_rows.append(
            {
                "arch": arch,
                "metric": "best_val_acc",
                "n_pairs": len(pairs),
                "true_mean": true_mean,
                "true_std": true_std,
                "shuffled_mean": shuffled_mean,
                "shuffled_std": shuffled_std,
                "mean_diff_true_minus_shuffled": float(diffs.mean()) if len(diffs) else float("nan"),
                "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                **test,
            }
        )

    write_csv(out_dir / "rotated_subject_metric_rows.csv", metric_rows)
    write_csv(out_dir / "rotated_subject_layer_rows.csv", layer_rows)
    write_csv(out_dir / "rotated_subject_arch_summary.csv", summary_rows)
    write_csv(out_dir / "rotated_subject_subject_summary.csv", subject_rows)
    write_csv(out_dir / "rotated_subject_true_vs_shuffled_tests.csv", test_rows)

    lines = [
        "# BCI IV 2a Rotated Held-Out Subject Robustness",
        "",
        "This robustness extension evaluates rotated held-out training subjects alongside the six-seed analysis.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'rotated_subject_metric_rows.csv'}`",
        f"- `{out_dir / 'rotated_subject_layer_rows.csv'}`",
        f"- `{out_dir / 'rotated_subject_arch_summary.csv'}`",
        f"- `{out_dir / 'rotated_subject_subject_summary.csv'}`",
        f"- `{out_dir / 'rotated_subject_true_vs_shuffled_tests.csv'}`",
        "",
        "## Architecture Summary",
        "",
        "| Architecture | Mode | n | Best validation accuracy |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(f"| `{row['arch']}` | {row['mode']} | {row['n']} | {row['best_val_acc_pm_std']} |")
    lines += [
        "",
        "## True-vs-Shuffled Held-Out Subject Tests",
        "",
        "| Architecture | n pairs | True mean | Shuffled mean | Diff | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        lines.append(
            f"| `{row['arch']}` | {row['n_pairs']} | {fmt(row['true_mean'])} | "
            f"{fmt(row['shuffled_mean'])} | {fmt(row['mean_diff_true_minus_shuffled'])} | "
            f"{fmt(row['p_value'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- These runs test whether BCI IV 2a conclusions survive rotating the held-out training subject instead of relying only on A09T.",
        "- This extension evaluates robustness across rotated held-out training subjects; its effect estimates are reported separately from the six-seed analysis.",
    ]
    (out_dir / "bci2a_rotated_subject_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
