#!/usr/bin/env python3
"""Audit paper-facing statistical coverage and add missing paired tests/CIs."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/statistical_completeness")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--paper-stats", default="reports/paper_statistics/all_true_vs_shuffled_corrected.csv")
    parser.add_argument("--supervised-tests", default="reports/multiseed_stats/true_vs_shuffled_wilcoxon.csv")
    parser.add_argument("--fm-tests", default="reports/fm_multiseed_stats/fm_true_vs_shuffled_wilcoxon.csv")
    parser.add_argument(
        "--supervised-corr-ci",
        default="reports/multiseed_stats/dynamics_correlation_bootstrap_ci.csv",
    )
    parser.add_argument(
        "--fbcsp-layer-metrics",
        default="reports/bci2a_fbcsp_dl_comparison/bci2a_fbcsp_dl_layer_metrics.csv",
    )
    parser.add_argument(
        "--filterbank-layer-metrics",
        default="reports/classical_filterbank_dl_comparison/classical_filterbank_dl_layer_metrics.csv",
    )
    parser.add_argument("--cka-correlations", default="reports/cka_rsa_stats/method_val_acc_correlations.csv")
    parser.add_argument(
        "--subject-cross-correlations",
        default="reports/bci2a_subject_odi_stats/cross_subject_odi_accuracy_correlations.csv",
    )
    parser.add_argument(
        "--subject-trajectory-correlations",
        default="reports/bci2a_subject_odi_stats/subject_odi_accuracy_trajectory_correlations.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    obs = min(w_plus, w_minus)
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


def add_effect_sizes(row: dict[str, Any], true_vals: list[float], shuffled_vals: list[float]) -> dict[str, Any]:
    diffs = np.asarray(true_vals, dtype=float) - np.asarray(shuffled_vals, dtype=float)
    mean_diff = float(np.mean(diffs)) if len(diffs) else float("nan")
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    dz = mean_diff / std_diff if std_diff > 0 else float("nan")
    n = int(row["n"])
    statistic = finite_float(row["statistic"], 0.0)
    total_rank = n * (n + 1) / 2.0
    sign = 1.0 if mean_diff >= 0 else -1.0
    rank_biserial = sign * (1.0 - 2.0 * statistic / total_rank) if total_rank > 0 else 0.0
    return {
        **row,
        "mean_diff_true_minus_shuffled": mean_diff,
        "std_diff": std_diff,
        "cohen_dz_paired": dz,
        "rank_biserial_signed": rank_biserial,
    }


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


def bh(p_values: list[float]) -> list[float]:
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


def classical_tests(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    by_key: dict[tuple[str, str, str, str, int], float] = {}
    for row in read_csv(Path(args.fbcsp_layer_metrics)):
        if row.get("family") != "fbcsp" or row.get("model") != "fbcsp":
            continue
        key = ("bci2a", "fbcsp", "fbcsp", row["mode"], int(row["seed"]))
        by_key[key] = finite_float(row["val_acc"])
    for row in read_csv(Path(args.filterbank_layer_metrics)):
        if row.get("family") != "classical_filterbank" or row.get("model") != "filterbank_logvar_lda":
            continue
        key = (row["dataset"], "classical_filterbank", row["model"], row["mode"], int(row["seed"]))
        by_key[key] = finite_float(row["val_acc"])

    groups = sorted({key[:3] for key in by_key})
    for dataset, family, model in groups:
        seeds = sorted(
            {
                key[4]
                for key in by_key
                if key[:3] == (dataset, family, model)
                and (dataset, family, model, "true", key[4]) in by_key
                and (dataset, family, model, "shuffled", key[4]) in by_key
            }
        )
        true_vals = [by_key[(dataset, family, model, "true", seed)] for seed in seeds]
        shuffled_vals = [by_key[(dataset, family, model, "shuffled", seed)] for seed in seeds]
        true_mean, true_std = mean_std(true_vals)
        shuffled_mean, shuffled_std = mean_std(shuffled_vals)
        row = {
            "dataset": dataset,
            "family": family,
            "model": model,
            "metric": "val_acc",
            "n_pairs": len(seeds),
            "true_mean": true_mean,
            "true_std": true_std,
            "shuffled_mean": shuffled_mean,
            "shuffled_std": shuffled_std,
            **wilcoxon_exact(true_vals, shuffled_vals),
        }
        rows.append(add_effect_sizes(row, true_vals, shuffled_vals))

    p_values = [finite_float(row["p_value"], 1.0) for row in rows]
    holm_values = holm(p_values)
    bh_values = bh(p_values)
    return [
        {
            **row,
            "p_holm_family": p_holm,
            "q_bh_family": p_bh,
            "significant_holm_0_05": p_holm <= 0.05,
            "significant_bh_0_05": p_bh <= 0.05,
        }
        for row, p_holm, p_bh in zip(rows, holm_values, bh_values)
    ]


def bootstrap_mean_ci(values_by_unit: dict[str, list[float]], n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    units = sorted(values_by_unit)
    values = [v for unit in units for v in values_by_unit[unit] if math.isfinite(v)]
    observed = float(np.mean(values)) if values else float("nan")
    if not units or not values:
        return observed, float("nan"), float("nan"), 0
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(units, size=len(units), replace=True)
        vals = [v for unit in sampled for v in values_by_unit[str(unit)] if math.isfinite(v)]
        if vals:
            boots.append(float(np.mean(vals)))
    lo, hi = np.percentile(np.asarray(boots), [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return observed, float(lo), float(hi), len(values)


def cka_bootstrap(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = read_csv(Path(args.cka_correlations))
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["dataset"], row["metric"])
        unit = f"{row['arch']}|{row['mode']}|{row['seed']}"
        grouped.setdefault(key, {}).setdefault(unit, []).append(finite_float(row["pearson_val_acc"]))
    out = []
    for (dataset, metric), values_by_unit in sorted(grouped.items()):
        mean, lo, hi, n = bootstrap_mean_ci(values_by_unit, args.bootstrap, rng)
        out.append(
            {
                "analysis": "cka_rsa_hsdd",
                "dataset": dataset,
                "metric": metric,
                "correlation": "pearson_val_acc",
                "mean": mean,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_rows": n,
                "bootstrap_unit": "arch_mode_seed",
                "bootstrap_n": args.bootstrap,
            }
        )
    return out


def subject_bootstrap(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    specs = [
        ("cross_subject", Path(args.subject_cross_correlations), "pearson_odi_accuracy"),
        ("within_subject_trajectory", Path(args.subject_trajectory_correlations), "pearson_odi_accuracy"),
    ]
    out = []
    for analysis, path, metric_col in specs:
        rows = read_csv(path)
        grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in rows:
            key = (analysis, row["arch"])
            unit = str(row["seed"])
            grouped.setdefault(key, {}).setdefault(unit, []).append(finite_float(row[metric_col]))
        for (analysis_name, arch), values_by_unit in sorted(grouped.items()):
            mean, lo, hi, n = bootstrap_mean_ci(values_by_unit, args.bootstrap, rng)
            out.append(
                {
                    "analysis": analysis_name,
                    "dataset": "bci2a",
                    "arch": arch,
                    "metric": metric_col,
                    "mean": mean,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_rows": n,
                    "bootstrap_unit": "seed",
                    "bootstrap_n": args.bootstrap,
                }
            )
    return out


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.3f}"


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    classical: list[dict[str, Any]],
    cka_ci: list[dict[str, Any]],
    subject_ci: list[dict[str, Any]],
) -> None:
    paper_rows = read_csv(Path(args.paper_stats))
    supervised_rows = read_csv(Path(args.supervised_tests))
    fm_rows = read_csv(Path(args.fm_tests))
    corr_rows = read_csv(Path(args.supervised_corr_ci))
    lines = [
        "# Statistical Completeness Audit",
        "",
        "This report separates primary inferential coverage from exploratory summaries.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'classical_true_vs_shuffled_effect_tests.csv'}`",
        f"- `{out_dir / 'cka_rsa_correlation_bootstrap_ci.csv'}`",
        f"- `{out_dir / 'subject_odi_correlation_bootstrap_ci.csv'}`",
        "",
        "## Coverage Status",
        "",
        "| Statistical family | Status | Notes |",
        "| --- | --- | --- |",
        f"| Supervised DL true-vs-shuffled paired tests | complete | {len(supervised_rows)} Wilcoxon rows; corrected/effect-size rows included in paper statistics. |",
        f"| Frozen-FM true-vs-shuffled paired tests | complete | {len(fm_rows)} Wilcoxon rows; corrected/effect-size rows included in paper statistics. |",
        f"| Combined paper corrected/effect-size table | complete | {len(paper_rows)} rows with Holm, BH, paired dz, and signed rank-biserial. |",
        f"| Supervised dynamics correlation CIs | complete | {len(corr_rows)} seed-trajectory bootstrap CI rows. |",
        f"| Classical EEG baselines | now complete | {len(classical)} paired true-vs-shuffled tests added here. |",
        f"| CKA/RSA vs HSDD correlation summaries | now CI-backed exploratory | {len(cka_ci)} bootstrap CI rows added here. |",
        f"| BCI subject-level ODI correlations | now CI-backed exploratory | {len(subject_ci)} bootstrap CI rows added here. |",
        "",
        "## Classical Baseline Paired Tests",
        "",
        "| Dataset | Model | True mean | Shuffled mean | Diff | p | Holm p | BH q | dz | Rank-biserial |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in classical:
        lines.append(
            f"| `{row['dataset']}` | `{row['model']}` | {fmt(row['true_mean'])} | "
            f"{fmt(row['shuffled_mean'])} | {fmt(row['mean_diff_true_minus_shuffled'])} | "
            f"{fmt(row['p_value'])} | {fmt(row['p_holm_family'])} | {fmt(row['q_bh_family'])} | "
            f"{fmt(row['cohen_dz_paired'])} | {fmt(row['rank_biserial_signed'])} |"
        )
    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- Validation accuracy true-vs-shuffled tests are the primary inferential contrasts.",
        "- Layerwise HSDD, CKA/RSA, sensitivity, and subject-ODI correlation tables are exploratory analyses; the compact contrast table states their analysis units and scope.",
        "- With six seeds, exact Wilcoxon p-values are intentionally coarse; effect sizes and confidence intervals should carry the weight of the interpretation.",
        "- FM statistics are complete only for frozen encoders/linear probes; fine-tuning and adapters are outside the current scope.",
    ]
    (out_dir / "statistical_completeness_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    classical = classical_tests(args)
    cka_ci = cka_bootstrap(args, rng)
    subject_ci = subject_bootstrap(args, rng)
    write_csv(out_dir / "classical_true_vs_shuffled_effect_tests.csv", classical)
    write_csv(out_dir / "cka_rsa_correlation_bootstrap_ci.csv", cka_ci)
    write_csv(out_dir / "subject_odi_correlation_bootstrap_ci.csv", subject_ci)
    write_summary(out_dir, args, classical, cka_ci, subject_ci)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
