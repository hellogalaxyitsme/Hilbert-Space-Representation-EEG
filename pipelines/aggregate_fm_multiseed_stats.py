#!/usr/bin/env python3
"""Aggregate frozen EEG-FM linear-probe multiseed reports."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


FMS = ("biot", "labram", "reve", "eegpt")
DEFAULT_DATASETS = ("bci2a", "sleepedf_full", "seediv", "p300")
MODES = ("true", "untrained", "shuffled")
DEFAULT_SEEDS = (41, 42, 43, 44, 45, 46)


def report_path(dataset: str, fm: str, mode: str, seed: int) -> Path:
    return Path(f"reports/{dataset}_{fm}_fm_feature_audit_{mode}_seed{seed}.json")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def rankdata_abs(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def wilcoxon_exact(x: list[float], y: list[float]) -> dict[str, float | int | str]:
    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    diffs = [diff for diff in diffs if abs(diff) > 1e-12]
    n = len(diffs)
    if n == 0:
        return {"test": "wilcoxon_signed_rank_exact", "n": 0, "statistic": 0.0, "p_value": 1.0}
    ranks = rankdata_abs([abs(diff) for diff in diffs])
    w_plus = float(sum(rank for rank, diff in zip(ranks, diffs) if diff > 0))
    w_minus = float(sum(rank for rank, diff in zip(ranks, diffs) if diff < 0))
    obs = min(w_plus, w_minus)
    totals = []
    for signs in itertools.product((0, 1), repeat=n):
        w = float(sum(rank for rank, sign in zip(ranks, signs) if sign))
        totals.append(min(w, float(sum(ranks) - w)))
    p = sum(total <= obs + 1e-12 for total in totals) / len(totals)
    return {"test": "wilcoxon_signed_rank_exact", "n": n, "statistic": obs, "p_value": float(p)}


def get_metric(payload: dict[str, Any], metric: str) -> float:
    cur: Any = payload
    for part in metric.split("."):
        cur = cur[part]
    if not isinstance(cur, (int, float)) or isinstance(cur, bool) or not math.isfinite(float(cur)):
        raise ValueError(metric)
    return float(cur)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def fmt_idx(
    idx: dict[tuple[str, str, str, str], dict[str, Any]],
    dataset: str,
    fm: str,
    mode: str,
    metric: str,
) -> str:
    row = idx.get((dataset, fm, mode, metric))
    if row is None:
        return "NA"
    return str(row["mean_pm_std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=" ".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--out-dir", default="reports/fm_multiseed_stats")
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.replace(",", " ").split() if item.strip())
    datasets = tuple(item for item in args.datasets.replace(",", " ").split() if item.strip())
    out_dir = Path(args.out_dir)

    payloads: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    missing = []
    for dataset in datasets:
        for fm in FMS:
            for mode in MODES:
                for seed in seeds:
                    path = report_path(dataset, fm, mode, seed)
                    if not path.exists():
                        missing.append(str(path))
                    else:
                        payloads[(dataset, fm, mode, seed)] = read_json(path)
    if missing:
        raise FileNotFoundError("missing reports:\n" + "\n".join(missing[:100]))

    metrics = {
        "best_val_acc": "training.best_val_acc",
        "final_val_acc": "training.final_val_acc",
        "final_train_acc": "training.final_train_acc",
        "feature_spearman": "layers.feature_identity.pairwise_isometry.distance_spearman",
        "feature_odi": "layers.feature_identity.band_orthogonality.output_mean_abs_offdiag_cosine_mean",
        "logits_spearman": "layers.logits.pairwise_isometry.distance_spearman",
        "logits_odi": "layers.logits.band_orthogonality.output_mean_abs_offdiag_cosine_mean",
    }

    raw_rows = []
    for (dataset, fm, mode, seed), payload in sorted(payloads.items()):
        for metric, dotted in metrics.items():
            try:
                value = get_metric(payload, dotted)
            except Exception:
                continue
            raw_rows.append(
                {
                    "dataset": dataset,
                    "fm": fm,
                    "mode": mode,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                    "source": str(report_path(dataset, fm, mode, seed)),
                }
            )

    summary_rows = []
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in raw_rows:
        key = (row["dataset"], row["fm"], row["mode"], row["metric"])
        grouped.setdefault(key, []).append(float(row["value"]))
    for (dataset, fm, mode, metric), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        summary_rows.append(
            {
                "dataset": dataset,
                "fm": fm,
                "mode": mode,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": std,
                "mean_pm_std": fmt(mean, std),
            }
        )

    test_rows = []
    raw_index = {
        (row["dataset"], row["fm"], row["mode"], row["seed"], row["metric"]): float(row["value"])
        for row in raw_rows
    }
    for dataset in datasets:
        for fm in FMS:
            for metric in metrics:
                pairs = []
                for seed in seeds:
                    tkey = (dataset, fm, "true", seed, metric)
                    skey = (dataset, fm, "shuffled", seed, metric)
                    if tkey in raw_index and skey in raw_index:
                        pairs.append((raw_index[tkey], raw_index[skey]))
                if len(pairs) < 2:
                    continue
                true_vals = [pair[0] for pair in pairs]
                shuf_vals = [pair[1] for pair in pairs]
                diffs = [a - b for a, b in pairs]
                mean_diff, std_diff = mean_std(diffs)
                test_rows.append(
                    {
                        "dataset": dataset,
                        "fm": fm,
                        "metric": metric,
                        "n_pairs": len(pairs),
                        "true_mean": mean_std(true_vals)[0],
                        "true_std": mean_std(true_vals)[1],
                        "shuffled_mean": mean_std(shuf_vals)[0],
                        "shuffled_std": mean_std(shuf_vals)[1],
                        "mean_diff_true_minus_shuffled": mean_diff,
                        "std_diff": std_diff,
                        **wilcoxon_exact(true_vals, shuf_vals),
                    }
                )

    write_csv(out_dir / "all_fm_seed_metric_values.csv", raw_rows, ["dataset", "fm", "mode", "seed", "metric", "value", "source"])
    write_csv(out_dir / "fm_mean_std_by_metric.csv", summary_rows, ["dataset", "fm", "mode", "metric", "n", "mean", "std", "mean_pm_std"])
    write_csv(out_dir / "fm_true_vs_shuffled_wilcoxon.csv", test_rows, ["dataset", "fm", "metric", "n_pairs", "true_mean", "true_std", "shuffled_mean", "shuffled_std", "mean_diff_true_minus_shuffled", "std_diff", "test", "n", "statistic", "p_value"])

    idx = {(row["dataset"], row["fm"], row["mode"], row["metric"]): row for row in summary_rows}
    test_idx = {(row["dataset"], row["fm"], row["metric"]): row for row in test_rows}
    lines = [
        "# EEG-FM Multiseed Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "",
        "Full metric tables:",
        "",
        f"- `{out_dir / 'all_fm_seed_metric_values.csv'}`",
        f"- `{out_dir / 'fm_mean_std_by_metric.csv'}`",
        f"- `{out_dir / 'fm_true_vs_shuffled_wilcoxon.csv'}`",
        "",
        "## Performance",
        "",
        "| Dataset | FM | True best val acc | Shuffled best val acc | Untrained-head best val acc | True-shuffled diff | p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for fm in FMS:
            true_row = idx[(dataset, fm, "true", "best_val_acc")]
            shuf_row = idx[(dataset, fm, "shuffled", "best_val_acc")]
            untrained_row = idx[(dataset, fm, "untrained", "best_val_acc")]
            test_row = test_idx[(dataset, fm, "best_val_acc")]
            lines.append(
                f"| {dataset} | {fm} | {true_row['mean_pm_std']} | {shuf_row['mean_pm_std']} | "
                f"{untrained_row['mean_pm_std']} | {test_row['mean_diff_true_minus_shuffled']:.3f} | {test_row['p_value']:.3f} |"
            )

    lines += [
        "",
        "## Frozen Representation Geometry",
        "",
        "| Dataset | FM | True feature ODI | Shuffled feature ODI | True feature Spearman | Shuffled feature Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for fm in FMS:
            lines.append(
                f"| {dataset} | {fm} | "
                f"{fmt_idx(idx, dataset, fm, 'true', 'feature_odi')} | "
                f"{fmt_idx(idx, dataset, fm, 'shuffled', 'feature_odi')} | "
                f"{fmt_idx(idx, dataset, fm, 'true', 'feature_spearman')} | "
                f"{fmt_idx(idx, dataset, fm, 'shuffled', 'feature_spearman')} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- These results use a frozen pretrained FM encoder with a linear probe. Full fine-tuning was outside this evaluation.",
        "- Sleep-EDF uses only two bipolar channels; the FM reports record each architecture's channel adaptation and should be treated as a montage-stress test.",
        "- Exact Wilcoxon p-values are paired by seed for true vs shuffled best validation accuracy.",
    ]
    (out_dir / "fm_multiseed_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
