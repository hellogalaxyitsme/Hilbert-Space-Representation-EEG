#!/usr/bin/env python3
"""Calculate multiple-comparison corrections and effect sizes for paired tests."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/paper_statistics")
    parser.add_argument("--multiseed", default="reports/multiseed_stats/true_vs_shuffled_wilcoxon.csv")
    parser.add_argument("--fm", default="reports/fm_multiseed_stats/fm_true_vs_shuffled_wilcoxon.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * p_values[idx]
        running = max(running, raw)
        adjusted[idx] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_high, idx in enumerate(order):
        rank = m - rank_from_high
        raw = p_values[idx] * m / rank
        running = min(running, raw)
        adjusted[idx] = min(1.0, running)
    return adjusted


def add_effects(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    out = []
    p_values = [finite_float(row.get("p_value"), 1.0) for row in rows]
    holm_values = holm(p_values)
    bh_values = benjamini_hochberg(p_values)
    for row, p_holm, p_bh in zip(rows, holm_values, bh_values):
        mean_diff = finite_float(row.get("mean_diff_true_minus_shuffled"))
        std_diff = finite_float(row.get("std_diff"))
        n = int(finite_float(row.get("n"), 0.0))
        statistic = finite_float(row.get("statistic"), 0.0)
        cohen_dz = mean_diff / std_diff if math.isfinite(mean_diff) and math.isfinite(std_diff) and std_diff > 0 else float("nan")
        total_rank = n * (n + 1) / 2.0
        sign = 1.0 if mean_diff >= 0 else -1.0
        rank_biserial = sign * (1.0 - 2.0 * statistic / total_rank) if n > 0 and total_rank > 0 else 0.0
        out.append(
            {
                **row,
                "test_family": family,
                "p_holm_family": p_holm,
                "q_bh_family": p_bh,
                "cohen_dz_paired": cohen_dz,
                "rank_biserial_signed": rank_biserial,
                "significant_holm_0_05": p_holm <= 0.05,
                "significant_bh_0_05": p_bh <= 0.05,
            }
        )
    return out


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    supervised = add_effects(read_csv(Path(args.multiseed)), "supervised_dl")
    fm = add_effects(read_csv(Path(args.fm)), "frozen_fm")
    combined = supervised + fm
    write_csv(out_dir / "supervised_true_vs_shuffled_corrected.csv", supervised)
    write_csv(out_dir / "fm_true_vs_shuffled_corrected.csv", fm)
    write_csv(out_dir / "all_true_vs_shuffled_corrected.csv", combined)

    focus = []
    for row in supervised:
        if row.get("metric") in {"training.best_val_acc", "history.best_val_acc", "history.final_val_acc"}:
            focus.append(row)
    for row in fm:
        if row.get("metric") in {"best_val_acc", "final_val_acc"}:
            focus.append(row)
    focus = sorted(
        focus,
        key=lambda r: (
            r.get("test_family", ""),
            r.get("dataset", ""),
            r.get("arch", r.get("fm", "")),
            r.get("experiment", ""),
            r.get("metric", ""),
        ),
    )
    write_csv(out_dir / "paper_focused_effect_tests.csv", focus)

    lines = [
        "# Statistical Corrections and Effect Sizes",
        "",
        "Corrections are applied within each test family: supervised DL metrics and frozen-FM metrics.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'supervised_true_vs_shuffled_corrected.csv'}`",
        f"- `{out_dir / 'fm_true_vs_shuffled_corrected.csv'}`",
        f"- `{out_dir / 'all_true_vs_shuffled_corrected.csv'}`",
        f"- `{out_dir / 'paper_focused_effect_tests.csv'}`",
        "",
        "Effect sizes:",
        "",
        "- `cohen_dz_paired`: mean paired true-shuffled difference divided by the paired-difference standard deviation.",
        "- `rank_biserial_signed`: signed Wilcoxon rank-biserial effect size; positive favors true labels.",
        "",
        "## Focused Performance Tests",
        "",
        "| Family | Dataset | Model | Experiment | Metric | True mean | Shuffled mean | Diff | p | Holm p | BH q | dz | Rank-biserial |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus:
        model = row.get("arch") or row.get("fm")
        experiment = row.get("experiment", "linear_probe")
        lines.append(
            f"| {row['test_family']} | `{row['dataset']}` | `{model}` | {experiment} | `{row['metric']}` | "
            f"{fmt(finite_float(row['true_mean']))} | {fmt(finite_float(row['shuffled_mean']))} | "
            f"{fmt(finite_float(row['mean_diff_true_minus_shuffled']))} | {fmt(finite_float(row['p_value']))} | "
            f"{fmt(finite_float(row['p_holm_family']))} | {fmt(finite_float(row['q_bh_family']))} | "
            f"{fmt(finite_float(row['cohen_dz_paired']))} | {fmt(finite_float(row['rank_biserial_signed']))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Exact Wilcoxon p-values remain useful for each planned paired true-vs-shuffled contrast, but corrected p/q values describe families of related tests.",
        "- With six seeds, the smallest possible two-sided exact Wilcoxon p-value is 0.03125, so Holm correction across large families is intentionally conservative.",
        "- Effect sizes are therefore essential: they show the magnitude and direction of the controlled contrast even when corrected p-values are conservative.",
    ]
    (out_dir / "paper_statistics_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
