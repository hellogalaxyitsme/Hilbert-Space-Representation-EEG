#!/usr/bin/env python3
"""Summarize Tier-3 BCI IV 2a lambda/schedule sensitivity diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", default="reports/tier3_regularizer_bci2a_lambda_schedule_stats")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        val = float(row.get(key, default))
    except Exception:
        return default
    return val if math.isfinite(val) else default


def fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def config_rows(rows: list[dict[str, Any]], *, arch: str = "all") -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("arch") == arch
        and (
            str(row.get("scope", "")).startswith("lambda_")
            or str(row.get("scope", "")).startswith("target_")
        )
    ]


def robustness_score(row: dict[str, Any]) -> float:
    return (
        f(row, "delta_band_noise_acc_arm4_minus_arm2_mean")
        + f(row, "delta_channel_dropout_acc_arm4_minus_arm2_mean")
        - max(0.0, f(row, "delta_clean_ece_arm4_minus_arm2_mean"))
    )


def main() -> None:
    args = parse_args()
    diagnostics_dir = Path(args.diagnostics_dir)
    endpoint = read_csv(diagnostics_dir / "tier3_diagnostic_endpoint_summary.csv")
    target = read_csv(diagnostics_dir / "tier3_posthoc_target_alignment_summary.csv")
    target_by_key = {(row["arch"], row["scope"]): row for row in target}

    rows = []
    for row in config_rows(endpoint):
        target_row = target_by_key.get((row["arch"], row["scope"]), {})
        rows.append(
            {
                "scope": row["scope"],
                "n": row["n"],
                "delta_band_noise_acc": f(row, "delta_band_noise_acc_arm4_minus_arm2_mean"),
                "delta_band_noise_acc_ci_low": f(row, "delta_band_noise_acc_arm4_minus_arm2_ci_low"),
                "delta_band_noise_acc_ci_high": f(row, "delta_band_noise_acc_arm4_minus_arm2_ci_high"),
                "delta_channel_dropout_acc": f(row, "delta_channel_dropout_acc_arm4_minus_arm2_mean"),
                "delta_channel_dropout_acc_ci_low": f(row, "delta_channel_dropout_acc_arm4_minus_arm2_ci_low"),
                "delta_channel_dropout_acc_ci_high": f(row, "delta_channel_dropout_acc_arm4_minus_arm2_ci_high"),
                "delta_ece": f(row, "delta_clean_ece_arm4_minus_arm2_mean"),
                "delta_target_error": f(target_row, "mean_delta_abs_error_to_taxon_arm4_minus_arm2_mean"),
                "fraction_layers_closer_to_taxon": f(target_row, "fraction_layers_arm4_closer_to_taxon_mean"),
                "delta_reg_to_task_ratio": f(row, "delta_regularizer_to_task_ratio_arm4_minus_arm2_mean"),
                "robustness_score": robustness_score(row),
            }
        )
    rows.sort(key=lambda r: r["robustness_score"], reverse=True)

    lines = [
        "# Tier-3 BCI IV 2a Regularizer Sensitivity Summary",
        "",
        "Primary question: does changing the HSDD regularizer strength, schedule, or target scheme make taxonomy-conditioned HSDD, arm 4, beat architecture-agnostic ODI minimization, arm 2?",
        "",
        "Positive robustness accuracy deltas favor arm 4; negative ECE and target-error deltas favor arm 4.",
        "",
    ]
    if rows:
        best = rows[0]
        lines += [
            "## Best Overall Configuration By Robustness Score",
            "",
            f"- Config: `{best['scope']}`",
            f"- Î” band-noise accuracy: {fmt(best['delta_band_noise_acc'])} 95% CI [{fmt(best['delta_band_noise_acc_ci_low'])}, {fmt(best['delta_band_noise_acc_ci_high'])}]",
            f"- Î” channel-dropout accuracy: {fmt(best['delta_channel_dropout_acc'])} 95% CI [{fmt(best['delta_channel_dropout_acc_ci_low'])}, {fmt(best['delta_channel_dropout_acc_ci_high'])}]",
            f"- Î” ECE: {fmt(best['delta_ece'])}",
            f"- Î” taxon-target absolute error: {fmt(best['delta_target_error'])}",
            f"- Fraction of layers closer to taxon target: {fmt(best['fraction_layers_closer_to_taxon'])}",
            "",
            "## All Overall Configurations",
            "",
            "| Config | n | Î” band-noise acc | Î” channel-drop acc | Î” ECE | Î” target error | Fraction closer | Î” reg/task | Score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| `{row['scope']}` | {row['n']} | {fmt(row['delta_band_noise_acc'])} | "
                f"{fmt(row['delta_channel_dropout_acc'])} | {fmt(row['delta_ece'])} | "
                f"{fmt(row['delta_target_error'])} | {fmt(row['fraction_layers_closer_to_taxon'])} | "
                f"{fmt(row['delta_reg_to_task_ratio'], 6)} | {fmt(row['robustness_score'])} |"
            )
    else:
        lines += [
            "No config-level rows were found. Check that `diagnose_tier3_bci2a_regularizer.py` was run on the sensitivity reports with lambda/schedule-aware pairing.",
        ]
    lines += [
        "",
        "Decision rule: only scale the regularizer to other datasets if at least one configuration gives positive primary robustness deltas for arm 4 over arm 2 and shows reliable movement toward taxon targets.",
    ]
    (diagnostics_dir / "tier3_lambda_schedule_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {diagnostics_dir / 'tier3_lambda_schedule_summary.md'}")


if __name__ == "__main__":
    main()
