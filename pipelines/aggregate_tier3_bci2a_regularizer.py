#!/usr/bin/env python3
"""Aggregate Tier-3 BCI IV 2a regularizer ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_ARMS = ("none", "odi_min", "weight_orth", "taxon_hsdd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/tier3_regularizer_bci2a")
    parser.add_argument("--out-dir", default="reports/tier3_regularizer_bci2a_stats")
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


def fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def bootstrap_ci(vals: list[float], *, n_boot: int = 5000, seed: int = 1729) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    rows = []
    band_rows = []
    channel_rows = []
    for path in sorted(reports_dir.glob("*.json")):
        payload = read_json(path)
        endpoints = payload["endpoints"]
        row = {
            "dataset": payload["dataset"]["name"],
            "arch": payload["model"]["arch"],
            "arm": payload["run"]["arm"],
            "seed": payload["run"]["seed"],
            "train_fraction": payload["dataset"]["train_fraction"],
            "epochs": payload["run"]["epochs"],
            "best_epoch_by_val_loss": payload["selection"]["best_epoch_by_val_loss"],
            "epochs_to_90pct_run_best_val_acc": payload["selection"]["epochs_to_90pct_run_best_val_acc"],
            "clean_acc": endpoints["clean"]["acc"],
            "clean_loss": endpoints["clean"]["loss"],
            "clean_ece": endpoints["clean"]["ece"],
            "band_noise_mean_acc": endpoints["band_limited_noise"]["mean_acc"],
            "band_noise_mean_acc_drop": endpoints["band_limited_noise"]["mean_acc_drop"],
            "band_noise_worst_acc": endpoints["band_limited_noise"]["worst_acc"],
            "band_noise_worst_acc_drop": endpoints["band_limited_noise"]["worst_acc_drop"],
            "channel_dropout_mean_acc": endpoints["channel_dropout"]["mean_acc"],
            "channel_dropout_mean_acc_drop": endpoints["channel_dropout"]["mean_acc_drop"],
            "channel_dropout_worst_acc": endpoints["channel_dropout"]["worst_acc"],
            "channel_dropout_worst_acc_drop": endpoints["channel_dropout"]["worst_acc_drop"],
            "source": str(path),
        }
        rows.append(row)
        for band in endpoints["band_limited_noise"]["rows"]:
            band_rows.append({**{k: row[k] for k in ("dataset", "arch", "arm", "seed", "train_fraction")}, **band})
        for drop in endpoints["channel_dropout"]["rows"]:
            channel_rows.append({**{k: row[k] for k in ("dataset", "arch", "arm", "seed", "train_fraction")}, **drop})

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["arch"], float(row["train_fraction"]))].append(row)

    arm_summary = []
    pair_rows = []
    for key, vals in sorted(grouped.items()):
        dataset, arch, train_fraction = key
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for val in vals:
            by_arm[val["arm"]].append(val)
        for arm in PRIMARY_ARMS:
            arm_vals = by_arm.get(arm, [])
            if not arm_vals:
                continue
            summary = {"dataset": dataset, "arch": arch, "train_fraction": train_fraction, "arm": arm, "n": len(arm_vals)}
            for metric in (
                "clean_acc",
                "clean_ece",
                "band_noise_mean_acc",
                "band_noise_mean_acc_drop",
                "channel_dropout_mean_acc",
                "channel_dropout_mean_acc_drop",
                "epochs_to_90pct_run_best_val_acc",
            ):
                metric_vals = [float(v[metric]) for v in arm_vals if v[metric] not in (None, "")]
                mean, std = mean_std(metric_vals)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            arm_summary.append(summary)

        odi = {int(v["seed"]): v for v in by_arm.get("odi_min", [])}
        taxon = {int(v["seed"]): v for v in by_arm.get("taxon_hsdd", [])}
        for seed in sorted(set(odi) & set(taxon)):
            a2 = odi[seed]
            a4 = taxon[seed]
            pair_rows.append(
                {
                    "dataset": dataset,
                    "arch": arch,
                    "train_fraction": train_fraction,
                    "seed": seed,
                    "delta_clean_acc_arm4_minus_arm2": float(a4["clean_acc"]) - float(a2["clean_acc"]),
                    "delta_clean_ece_arm4_minus_arm2": float(a4["clean_ece"]) - float(a2["clean_ece"]),
                    "delta_band_noise_acc_arm4_minus_arm2": float(a4["band_noise_mean_acc"]) - float(a2["band_noise_mean_acc"]),
                    "delta_band_noise_drop_arm4_minus_arm2": float(a4["band_noise_mean_acc_drop"]) - float(a2["band_noise_mean_acc_drop"]),
                    "delta_channel_dropout_acc_arm4_minus_arm2": float(a4["channel_dropout_mean_acc"]) - float(a2["channel_dropout_mean_acc"]),
                    "delta_channel_dropout_drop_arm4_minus_arm2": float(a4["channel_dropout_mean_acc_drop"]) - float(a2["channel_dropout_mean_acc_drop"]),
                }
            )

    comparison_summary = []
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_scope[("all", "all")].append(row)
        by_scope[(row["arch"], "all_fractions")].append(row)
        by_scope[(row["arch"], f"fraction_{row['train_fraction']:g}")].append(row)
    for (arch, scope), vals in sorted(by_scope.items()):
        out = {"arch": arch, "scope": scope, "n_pairs": len(vals)}
        for metric in (
            "delta_band_noise_acc_arm4_minus_arm2",
            "delta_band_noise_drop_arm4_minus_arm2",
            "delta_channel_dropout_acc_arm4_minus_arm2",
            "delta_channel_dropout_drop_arm4_minus_arm2",
            "delta_clean_ece_arm4_minus_arm2",
            "delta_clean_acc_arm4_minus_arm2",
        ):
            metric_vals = [float(v[metric]) for v in vals]
            mean, std = mean_std(metric_vals)
            lo, hi = bootstrap_ci(metric_vals)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci_low"] = lo
            out[f"{metric}_ci_high"] = hi
        comparison_summary.append(out)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "tier3_regularizer_run_rows.csv", rows)
    write_csv(out_dir / "tier3_regularizer_band_noise_rows.csv", band_rows)
    write_csv(out_dir / "tier3_regularizer_channel_dropout_rows.csv", channel_rows)
    write_csv(out_dir / "tier3_regularizer_arm_summary.csv", arm_summary)
    write_csv(out_dir / "tier3_arm4_vs_arm2_pairs.csv", pair_rows)
    write_csv(out_dir / "tier3_arm4_vs_arm2_summary.csv", comparison_summary)

    lines = [
        "# Tier-3 BCI IV 2a HSDD Regularizer Ablation",
        "",
        "Primary comparison: taxonomy-conditioned HSDD regularizer (arm 4) versus architecture-agnostic ODI minimization (arm 2). Positive accuracy deltas and negative robustness-drop/ECE deltas favor arm 4.",
        "",
        "## Coverage",
        "",
        f"- Completed run JSONs: {len(rows)}",
        f"- Arm-4-vs-arm-2 paired comparisons: {len(pair_rows)}",
        "",
        "## Arm 4 vs Arm 2 Summary",
        "",
        "| Architecture | Scope | n | Î” band-noise acc | Î” band-noise drop | Î” channel-drop acc | Î” channel-drop drop | Î” ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_summary:
        lines.append(
            f"| `{row['arch']}` | `{row['scope']}` | {row['n_pairs']} | "
            f"{fmt(row['delta_band_noise_acc_arm4_minus_arm2_mean'])} "
            f"[{fmt(row['delta_band_noise_acc_arm4_minus_arm2_ci_low'])}, {fmt(row['delta_band_noise_acc_arm4_minus_arm2_ci_high'])}] | "
            f"{fmt(row['delta_band_noise_drop_arm4_minus_arm2_mean'])} "
            f"[{fmt(row['delta_band_noise_drop_arm4_minus_arm2_ci_low'])}, {fmt(row['delta_band_noise_drop_arm4_minus_arm2_ci_high'])}] | "
            f"{fmt(row['delta_channel_dropout_acc_arm4_minus_arm2_mean'])} "
            f"[{fmt(row['delta_channel_dropout_acc_arm4_minus_arm2_ci_low'])}, {fmt(row['delta_channel_dropout_acc_arm4_minus_arm2_ci_high'])}] | "
            f"{fmt(row['delta_channel_dropout_drop_arm4_minus_arm2_mean'])} "
            f"[{fmt(row['delta_channel_dropout_drop_arm4_minus_arm2_ci_low'])}, {fmt(row['delta_channel_dropout_drop_arm4_minus_arm2_ci_high'])}] | "
            f"{fmt(row['delta_clean_ece_arm4_minus_arm2_mean'])} "
            f"[{fmt(row['delta_clean_ece_arm4_minus_arm2_ci_low'])}, {fmt(row['delta_clean_ece_arm4_minus_arm2_ci_high'])}] |"
        )
    lines += [
        "",
        "Interpretation rule: if arm 4 improves primary robustness endpoints over arm 2, the architecture taxonomy has causal support. If arm 4 does not beat arm 2, the taxonomy should be described as correlational.",
    ]
    (out_dir / "tier3_regularizer_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
