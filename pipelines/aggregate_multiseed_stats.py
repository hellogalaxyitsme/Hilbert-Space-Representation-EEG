#!/usr/bin/env python3
"""Aggregate multiseed HSDD experiments with paired tests and bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
DEFAULT_DATASETS = ("bci2a", "sleepedf_full", "seediv", "p300")
DEFAULT_SEEDS = (41, 42, 43, 44, 45, 46)
STATIC_MODES = ("true", "untrained", "shuffled")
DYNAMIC_MODES = ("true", "shuffled")


BCI_TRUE_BASE = {
    "eegnet": "reports/eegnet_bci2a_layer_audit.json",
    "shallowconvnet": "reports/shallowconvnet_bci2a_layer_audit.json",
    "eegconformer": "reports/eegconformer_bci2a_layer_audit.json",
    "tsception": "reports/tsception_bci2a_layer_audit_true.json",
    "atcnet": "reports/atcnet_bci2a_layer_audit_true.json",
}


def static_path(dataset: str, arch: str, mode: str, seed: int) -> Path:
    if dataset == "bci2a":
        if seed == 41:
            if mode == "true":
                return Path(BCI_TRUE_BASE[arch])
            return Path(f"reports/{arch}_bci2a_layer_audit_{mode}.json")
        return Path(f"reports/{arch}_bci2a_layer_audit_{mode}_seed{seed}.json")
    if dataset == "seediv":
        return Path(f"reports/seediv_{arch}_layer_audit_{mode}_seed{seed}.json")
    if dataset == "p300":
        return Path(f"reports/p300_{arch}_layer_audit_{mode}_seed{seed}.json")
    if dataset == "sleepedf" and seed == 41:
        return Path(f"reports/sleepedf_{arch}_layer_audit_{mode}.json")
    if dataset in {"sleepedf", "sleepedf_full"}:
        return Path(f"reports/{dataset}_{arch}_layer_audit_{mode}_seed{seed}.json")
    raise ValueError(f"unsupported dataset: {dataset}")


def dynamics_path(dataset: str, arch: str, mode: str, seed: int) -> Path:
    if dataset == "bci2a":
        prefix = f"{arch}_bci2a"
        if seed == 41:
            return Path(f"reports/{prefix}_dynamics_{mode}.json")
    elif dataset == "seediv":
        prefix = f"seediv_{arch}"
    elif dataset == "p300":
        prefix = f"p300_{arch}"
    elif dataset == "sleepedf":
        prefix = f"sleepedf_{arch}"
        if seed == 41:
            return Path(f"reports/{prefix}_dynamics_{mode}.json")
    elif dataset == "sleepedf_full":
        prefix = f"sleepedf_full_{arch}"
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return Path(f"reports/{prefix}_dynamics_{mode}_seed{seed}.json")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_scalars(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if is_number(obj):
        out[prefix] = float(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_scalars(value, child))
    elif isinstance(obj, list):
        # Avoid flattening matrices/vectors; named checkpoint summaries are handled separately.
        return out
    return out


def static_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = flatten_scalars({"training": payload.get("training", {}), "layers": payload.get("layers", {})})
    return {
        k: v
        for k, v in metrics.items()
        if not k.endswith(".n_pairs")
        and not k.endswith(".n_epochs")
        and not k.endswith(".n_points")
        and not k.endswith(".n_directions")
        and not k.endswith(".eps")
    }


def dynamics_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    history = payload.get("history", [])
    if history:
        metrics["history.best_val_acc"] = max(float(row["val_acc"]) for row in history)
        metrics["history.final_val_acc"] = float(history[-1]["val_acc"])
        metrics["history.final_train_acc"] = float(history[-1]["train_acc"])
        metrics["history.final_train_loss"] = float(history[-1]["train_loss"])
        metrics["history.final_val_loss"] = float(history[-1]["val_loss"])
    for audit in payload.get("audits", []):
        epoch = int(audit["epoch"])
        metrics[f"audit_epoch_{epoch}.train_acc"] = float(audit["train_acc"])
        metrics[f"audit_epoch_{epoch}.val_acc"] = float(audit["val_acc"])
        for layer, vals in audit.get("compact", {}).items():
            for name, value in vals.items():
                if is_number(value):
                    metrics[f"audit_epoch_{epoch}.compact.{layer}.{name}"] = float(value)
    if payload.get("audits"):
        final = payload["audits"][-1]
        for layer, vals in final.get("compact", {}).items():
            for name, value in vals.items():
                if is_number(value):
                    metrics[f"final_compact.{layer}.{name}"] = float(value)
    return metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


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


def wilcoxon_exact(x: list[float], y: list[float]) -> dict[str, float | str | int]:
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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_corr_ci(seed_points: dict[int, list[tuple[float, float]]], n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    seeds = sorted(seed_points)
    x_all = np.asarray([p[0] for seed in seeds for p in seed_points[seed]], dtype=float)
    y_all = np.asarray([p[1] for seed in seeds for p in seed_points[seed]], dtype=float)
    observed = pearson(x_all, y_all)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(seeds, size=len(seeds), replace=True)
        pts = [p for seed in sample for p in seed_points[int(seed)]]
        xb = np.asarray([p[0] for p in pts], dtype=float)
        yb = np.asarray([p[1] for p in pts], dtype=float)
        rb = pearson(xb, yb)
        if math.isfinite(rb):
            boots.append(rb)
    if not boots:
        return observed, float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(boots), [2.5, 97.5])
    return observed, float(lo), float(hi)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/multiseed_stats")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--seeds",
        default=" ".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma or space separated random seeds to aggregate.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    seeds = tuple(int(item) for item in args.seeds.replace(",", " ").split() if item.strip())
    datasets = tuple(item for item in args.datasets.replace(",", " ").split() if item.strip())
    if len(seeds) < 2:
        raise ValueError("--seeds must contain at least two seeds")

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    payload_cache: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}

    for dataset in datasets:
        for arch in ARCHES:
            for mode in STATIC_MODES:
                for seed in seeds:
                    path = static_path(dataset, arch, mode, seed)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    payload = read_json(path)
                    payload_cache[(dataset, arch, "static", mode, seed)] = payload
                    for metric, value in static_metrics(payload).items():
                        records.append({
                            "dataset": dataset,
                            "arch": arch,
                            "experiment": "static",
                            "condition": mode,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                            "source": str(path),
                        })
            for mode in DYNAMIC_MODES:
                for seed in seeds:
                    path = dynamics_path(dataset, arch, mode, seed)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    payload = read_json(path)
                    payload_cache[(dataset, arch, "dynamics", mode, seed)] = payload
                    for metric, value in dynamics_metrics(payload).items():
                        records.append({
                            "dataset": dataset,
                            "arch": arch,
                            "experiment": "dynamics",
                            "condition": mode,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                            "source": str(path),
                        })

    if missing:
        raise FileNotFoundError("missing expected reports:\n" + "\n".join(missing[:100]))

    summary_rows = []
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in records:
        key = (row["dataset"], row["arch"], row["experiment"], row["condition"], row["metric"])
        grouped.setdefault(key, []).append(float(row["value"]))
    for (dataset, arch, experiment, condition, metric), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        summary_rows.append({
            "dataset": dataset,
            "arch": arch,
            "experiment": experiment,
            "condition": condition,
            "metric": metric,
            "n": len(values),
            "mean": mean,
            "std": std,
            "mean_pm_std": fmt_mean_std(mean, std),
        })

    test_rows = []
    for dataset in datasets:
        for arch in ARCHES:
            for experiment in ("static", "dynamics"):
                true_rows = {
                    (row["seed"], row["metric"]): float(row["value"])
                    for row in records
                    if row["dataset"] == dataset and row["arch"] == arch and row["experiment"] == experiment and row["condition"] == "true"
                }
                shuffled_rows = {
                    (row["seed"], row["metric"]): float(row["value"])
                    for row in records
                    if row["dataset"] == dataset and row["arch"] == arch and row["experiment"] == experiment and row["condition"] == "shuffled"
                }
                metrics = sorted({metric for _, metric in true_rows} & {metric for _, metric in shuffled_rows})
                for metric in metrics:
                    pairs = [(true_rows[(seed, metric)], shuffled_rows[(seed, metric)]) for seed in seeds if (seed, metric) in true_rows and (seed, metric) in shuffled_rows]
                    if len(pairs) < 2:
                        continue
                    true_vals = [p[0] for p in pairs]
                    shuf_vals = [p[1] for p in pairs]
                    diff = [a - b for a, b in pairs]
                    mean_diff, std_diff = mean_std(diff)
                    test = wilcoxon_exact(true_vals, shuf_vals)
                    test_rows.append({
                        "dataset": dataset,
                        "arch": arch,
                        "experiment": experiment,
                        "metric": metric,
                        "n_pairs": len(pairs),
                        "true_mean": mean_std(true_vals)[0],
                        "true_std": mean_std(true_vals)[1],
                        "shuffled_mean": mean_std(shuf_vals)[0],
                        "shuffled_std": mean_std(shuf_vals)[1],
                        "mean_diff_true_minus_shuffled": mean_diff,
                        "std_diff": std_diff,
                        **test,
                    })

    corr_rows = []
    for dataset in datasets:
        for arch in ARCHES:
            for condition in DYNAMIC_MODES:
                layer_names = set()
                for seed in seeds:
                    payload = payload_cache[(dataset, arch, "dynamics", condition, seed)]
                    for audit in payload.get("audits", []):
                        layer_names.update(audit.get("compact", {}).keys())
                for layer in sorted(layer_names):
                    for geom_metric in ("odi", "spearman"):
                        seed_points: dict[int, list[tuple[float, float]]] = {}
                        for seed in seeds:
                            payload = payload_cache[(dataset, arch, "dynamics", condition, seed)]
                            pts = []
                            for audit in payload.get("audits", []):
                                compact = audit.get("compact", {})
                                if layer in compact and geom_metric in compact[layer]:
                                    pts.append((float(audit["val_acc"]), float(compact[layer][geom_metric])))
                            if pts:
                                seed_points[seed] = pts
                        if len(seed_points) >= 2:
                            observed, lo, hi = bootstrap_corr_ci(seed_points, args.bootstrap, rng)
                            corr_rows.append({
                                "dataset": dataset,
                                "arch": arch,
                                "condition": condition,
                                "layer": layer,
                                "geometry_metric": geom_metric,
                                "correlation": "pearson_val_acc_vs_geometry",
                                "r": observed,
                                "bootstrap_ci_low": lo,
                                "bootstrap_ci_high": hi,
                                "bootstrap_n": args.bootstrap,
                                "bootstrap_unit": "seed_trajectory",
                            })

    write_csv(out_dir / "all_seed_metric_values.csv", records, ["dataset", "arch", "experiment", "condition", "seed", "metric", "value", "source"])
    write_csv(out_dir / "mean_std_by_metric.csv", summary_rows, ["dataset", "arch", "experiment", "condition", "metric", "n", "mean", "std", "mean_pm_std"])
    write_csv(out_dir / "true_vs_shuffled_wilcoxon.csv", test_rows, ["dataset", "arch", "experiment", "metric", "n_pairs", "true_mean", "true_std", "shuffled_mean", "shuffled_std", "mean_diff_true_minus_shuffled", "std_diff", "test", "n", "statistic", "p_value"])
    write_csv(out_dir / "dynamics_correlation_bootstrap_ci.csv", corr_rows, ["dataset", "arch", "condition", "layer", "geometry_metric", "correlation", "r", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_n", "bootstrap_unit"])

    key_metrics = [
        ("static", "training.best_val_acc"),
        ("static", "training.final_train_acc"),
        ("dynamics", "history.best_val_acc"),
        ("dynamics", "history.final_val_acc"),
    ]
    lines = [
        "# Multiseed Statistical Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "",
        "Full metric tables:",
        "",
        f"- `{out_dir / 'all_seed_metric_values.csv'}`",
        f"- `{out_dir / 'mean_std_by_metric.csv'}`",
        f"- `{out_dir / 'true_vs_shuffled_wilcoxon.csv'}`",
        f"- `{out_dir / 'dynamics_correlation_bootstrap_ci.csv'}`",
        "",
        "Wilcoxon signed-rank tests are exact and paired by seed. Correlation CIs use seed-trajectory block bootstrap.",
        "",
        "## Key Performance Metrics",
        "",
        "| Dataset | Architecture | Experiment | Condition | Metric | Mean +/- std |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    summary_index = {
        (row["dataset"], row["arch"], row["experiment"], row["condition"], row["metric"]): row
        for row in summary_rows
    }
    for dataset in datasets:
        for arch in ARCHES:
            for experiment, metric in key_metrics:
                conditions = STATIC_MODES if experiment == "static" else DYNAMIC_MODES
                for condition in conditions:
                    row = summary_index.get((dataset, arch, experiment, condition, metric))
                    if row:
                        lines.append(f"| {dataset} | {arch} | {experiment} | {condition} | `{metric}` | {row['mean_pm_std']} |")
    lines += [
        "",
        "## Key True vs Shuffled Tests",
        "",
        "| Dataset | Architecture | Experiment | Metric | True mean | Shuffled mean | Mean diff | p |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    wanted_tests = {
        ("static", "training.best_val_acc"),
        ("dynamics", "history.best_val_acc"),
        ("dynamics", "history.final_val_acc"),
    }
    for row in test_rows:
        if (row["experiment"], row["metric"]) in wanted_tests:
            lines.append(
                f"| {row['dataset']} | {row['arch']} | {row['experiment']} | `{row['metric']}` | "
                f"{row['true_mean']:.3f} | {row['shuffled_mean']:.3f} | "
                f"{row['mean_diff_true_minus_shuffled']:.3f} | {row['p_value']:.3f} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- With six seeds, exact paired Wilcoxon p-values can resolve one-sided-consistent effects to p = 0.03125; effect sizes and mean +/- std still carry the main interpretive weight.",
        "- Full layer-wise HSDD metric means, standard deviations, and tests are in the CSV artifacts.",
    ]
    (out_dir / "multiseed_statistical_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
