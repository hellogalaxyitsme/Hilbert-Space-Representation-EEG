#!/usr/bin/env python3
"""Utility demo: HSDD-guided early model selection.

The demo asks a practical question:

Given only early-checkpoint geometry summaries, can we select an architecture
for a dataset/seed and compare its future validation accuracy with an oracle reference?

It compares HSDD feature sets against CKA/RSA and early validation accuracy.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


KEYS = ("dataset", "arch", "mode", "seed", "epoch")
HSDD_METRICS = (
    "hsdd_distance_spearman",
    "hsdd_scaled_ratio_error_mean",
    "hsdd_odi",
    "hsdd_collapse_factor",
)
CKA_RSA_METRICS = (
    "cka_input",
    "rsa_input_spearman",
    "rsa_label_spearman",
    "activation_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined",
        default="reports/cka_rsa_stats/joined_layer_checkpoint_metrics.csv",
        help="Joined HSDD/CKA/RSA layer-checkpoint table.",
    )
    parser.add_argument("--out-dir", default="reports/utility_demo_model_selection")
    parser.add_argument("--mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--epochs", default="0 1 2 5")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--protocols",
        default="leave_dataset_seed_out leave_dataset_out",
        help="Evaluation protocols to run.",
    )
    return parser.parse_args()


def parse_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [v - factor * base for v, base in zip(aug[row], aug[col])]
    return [aug[i][-1] for i in range(n)]


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.means: list[float] = []
        self.scales: list[float] = []
        self.coef: list[float] = []

    def fit(self, x: list[list[float]], y: list[float]) -> None:
        if not x:
            raise ValueError("no training rows")
        n_features = len(x[0])
        self.means = [mean(row[j] for row in x) for j in range(n_features)]
        self.scales = []
        x_std = []
        for row in x:
            x_std.append([])
            for j, value in enumerate(row):
                # Scale is filled below once all columns have been inspected.
                x_std[-1].append(value)
        self.scales = []
        for j in range(n_features):
            col = [row[j] for row in x]
            scale = std(col)
            self.scales.append(scale if scale > 1e-12 else 1.0)
        design = [[1.0] + [(value - self.means[j]) / self.scales[j] for j, value in enumerate(row)] for row in x]
        p = n_features + 1
        xtx = [[0.0 for _ in range(p)] for _ in range(p)]
        xty = [0.0 for _ in range(p)]
        for row, target in zip(design, y):
            for i in range(p):
                xty[i] += row[i] * target
                for j in range(p):
                    xtx[i][j] += row[i] * row[j]
        for i in range(1, p):
            xtx[i][i] += self.alpha
        self.coef = solve_linear_system(xtx, xty)

    def predict_one(self, row: list[float]) -> float:
        vals = [1.0] + [(value - self.means[j]) / self.scales[j] for j, value in enumerate(row)]
        return dot(vals, self.coef)


def read_joined(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["mode"] == mode]
    if not rows:
        raise ValueError(f"no rows found for mode={mode}")
    return rows


def aggregate_layer_rows(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, str, int], dict[str, float]], dict[tuple[str, str, str, int], dict[str, float]]]:
    layer_values: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    val_acc: dict[tuple[str, str, str, int], float] = {}

    for row in rows:
        key = (row["dataset"], row["arch"], row["seed"], int(row["epoch"]))
        val = parse_float(row.get("val_acc"))
        if val is not None:
            val_acc[key] = val
        for metric in HSDD_METRICS + CKA_RSA_METRICS:
            parsed = parse_float(row.get(metric))
            if parsed is not None:
                layer_values[key][metric].append(parsed)

    features: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for key, by_metric in layer_values.items():
        out: dict[str, float] = {}
        for metric, vals in by_metric.items():
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = std(vals)
            out[f"{metric}_min"] = min(vals)
            out[f"{metric}_max"] = max(vals)
            out[f"{metric}_abs_mean"] = mean(abs(v) for v in vals)
        features[key] = out

    for key, out in list(features.items()):
        epoch0_key = (key[0], key[1], key[2], 0)
        base = features.get(epoch0_key)
        if not base:
            continue
        for name, value in list(out.items()):
            if name in base:
                out[f"delta_{name}"] = value - base[name]

    targets: dict[tuple[str, str, str, int], dict[str, float]] = {}
    by_run_arch: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for key, value in val_acc.items():
        dataset, arch, seed, epoch = key
        by_run_arch[(dataset, arch, seed)].append((epoch, value))

    for key, current_val in val_acc.items():
        dataset, arch, seed, epoch = key
        future = [value for e, value in by_run_arch[(dataset, arch, seed)] if e >= epoch]
        all_vals = [value for _e, value in by_run_arch[(dataset, arch, seed)]]
        targets[key] = {
            "current_val_acc": current_val,
            "future_best_val_acc": max(future),
            "all_checkpoint_best_val_acc": max(all_vals),
            "last_checkpoint_val_acc": sorted(by_run_arch[(dataset, arch, seed)])[-1][1],
        }
    return features, targets


def feature_names(kind: str, available: Iterable[str]) -> list[str]:
    names = sorted(set(available))
    if kind == "early_val":
        return ["current_val_acc"]
    if kind == "hsdd":
        return [n for n in names if n.startswith("hsdd_") or n.startswith("delta_hsdd_")]
    if kind == "cka_rsa":
        return [
            n
            for n in names
            if n.startswith("cka_")
            or n.startswith("rsa_")
            or n.startswith("activation_rank")
            or n.startswith("delta_cka_")
            or n.startswith("delta_rsa_")
            or n.startswith("delta_activation_rank")
        ]
    if kind == "hsdd_cka_rsa":
        return feature_names("hsdd", names) + feature_names("cka_rsa", names)
    if kind == "early_val_hsdd":
        return ["current_val_acc"] + feature_names("hsdd", names)
    if kind == "all":
        return ["current_val_acc"] + feature_names("hsdd_cka_rsa", names)
    raise ValueError(kind)


def vectorize(row: dict[str, float], names: list[str]) -> list[float]:
    return [float(row.get(name, 0.0)) if math.isfinite(float(row.get(name, 0.0))) else 0.0 for name in names]


def split_allowed(protocol: str, train_key: tuple[str, str], test_key: tuple[str, str]) -> bool:
    train_dataset, train_seed = train_key
    test_dataset, test_seed = test_key
    if protocol == "leave_dataset_seed_out":
        return train_key != test_key
    if protocol == "leave_dataset_out":
        return train_dataset != test_dataset
    raise ValueError(protocol)


def evaluate_selector(
    *,
    features: dict[tuple[str, str, str, int], dict[str, float]],
    targets: dict[tuple[str, str, str, int], dict[str, float]],
    epoch: int,
    kind: str,
    protocol: str,
    ridge: float,
) -> tuple[list[dict[str, object]], list[str]]:
    keys = [key for key in features if key[3] == epoch and key in targets]
    if not keys:
        return [], []

    all_feature_names = set()
    for key in keys:
        all_feature_names.update(features[key])
    names = feature_names(kind, all_feature_names)
    if kind != "early_val" and not names:
        return [], []

    rows = []
    groups = sorted({(dataset, seed) for dataset, _arch, seed, _epoch in keys})
    for dataset, seed in groups:
        candidate_keys = [key for key in keys if key[0] == dataset and key[2] == seed]
        if len(candidate_keys) < 2:
            continue

        if kind == "early_val":
            predictions = {key: targets[key]["current_val_acc"] for key in candidate_keys}
        else:
            train_x: list[list[float]] = []
            train_y: list[float] = []
            for key in keys:
                train_group = (key[0], key[2])
                if not split_allowed(protocol, train_group, (dataset, seed)):
                    continue
                merged = {**features[key], **targets[key]}
                train_x.append(vectorize(merged, names))
                train_y.append(targets[key]["future_best_val_acc"])
            if len(train_x) <= len(names):
                continue
            model = RidgeRegressor(alpha=ridge)
            model.fit(train_x, train_y)
            predictions = {}
            for key in candidate_keys:
                merged = {**features[key], **targets[key]}
                predictions[key] = model.predict_one(vectorize(merged, names))

        selected = max(candidate_keys, key=lambda key: predictions[key])
        oracle = max(candidate_keys, key=lambda key: targets[key]["future_best_val_acc"])
        selected_target = targets[selected]["future_best_val_acc"]
        oracle_target = targets[oracle]["future_best_val_acc"]
        random_expected = mean(targets[key]["future_best_val_acc"] for key in candidate_keys)
        rows.append(
            {
                "protocol": protocol,
                "feature_set": kind,
                "epoch": epoch,
                "dataset": dataset,
                "seed": seed,
                "selected_arch": selected[1],
                "oracle_arch": oracle[1],
                "selected_future_best_val_acc": selected_target,
                "oracle_future_best_val_acc": oracle_target,
                "random_expected_future_best_val_acc": random_expected,
                "regret": oracle_target - selected_target,
                "hit": int(selected[1] == oracle[1]),
                "n_candidates": len(candidate_keys),
            }
        )
    return rows, names


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["feature_set"]), int(row["epoch"]))].append(row)

    out = []
    for (protocol, feature_set, epoch), vals in sorted(grouped.items()):
        selected = [float(row["selected_future_best_val_acc"]) for row in vals]
        oracle = [float(row["oracle_future_best_val_acc"]) for row in vals]
        random_expected = [float(row["random_expected_future_best_val_acc"]) for row in vals]
        regrets = [float(row["regret"]) for row in vals]
        hits = [int(row["hit"]) for row in vals]
        out.append(
            {
                "protocol": protocol,
                "feature_set": feature_set,
                "epoch": epoch,
                "n_decisions": len(vals),
                "mean_selected_future_best_val_acc": mean(selected),
                "mean_oracle_future_best_val_acc": mean(oracle),
                "mean_random_expected_future_best_val_acc": mean(random_expected),
                "mean_regret": mean(regrets),
                "median_regret": statistics.median(regrets) if regrets else float("nan"),
                "p75_regret": quantile(regrets, 0.75),
                "hit_rate": mean(hits),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.3f}"


def write_markdown(path: Path, summary: list[dict[str, object]], feature_counts: dict[tuple[str, int], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HSDD-Guided Model Selection Demo",
        "",
        "Task: choose one architecture per dataset/seed at an early checkpoint, using only early diagnostics, and compare the chosen architecture's future best validation accuracy against the oracle architecture for that same dataset/seed.",
        "",
        "Feature sets:",
        "",
        "- `hsdd`: layer-aggregated HSDD distance, scaled distortion, ODI, collapse-factor summaries and epoch-0 deltas.",
        "- `cka_rsa`: layer-aggregated CKA/input-RSA/label-RSA/rank summaries and epoch-0 deltas.",
        "- `early_val`: validation accuracy at the same early checkpoint.",
        "- `early_val_hsdd`: early validation accuracy plus HSDD features.",
        "- `all`: early validation accuracy plus HSDD, CKA, and RSA.",
        "",
        "Protocols:",
        "",
        "- `leave_dataset_seed_out`: train the selector on all other dataset/seed groups.",
        "- `leave_dataset_out`: train the selector on other datasets only, then transfer to the held-out dataset.",
        "",
        "## Summary",
        "",
        "| Protocol | Epoch | Feature set | Decisions | Selected acc | Oracle acc | Random acc | Regret | Hit rate |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['protocol']} | {row['epoch']} | `{row['feature_set']}` | {row['n_decisions']} | "
            f"{fmt(row['mean_selected_future_best_val_acc'])} | {fmt(row['mean_oracle_future_best_val_acc'])} | "
            f"{fmt(row['mean_random_expected_future_best_val_acc'])} | {fmt(row['mean_regret'])} | "
            f"{fmt(row['hit_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare selected, random, and oracle-reference accuracies together with regret and hit rate.",
            "- Differences between `hsdd` and `cka_rsa` summarize the information carried by their respective feature sets.",
            "- Differences between `early_val_hsdd` and `early_val` quantify the contribution of the added geometry features at the selected checkpoint.",
            "- The stricter `leave_dataset_out` protocol is the transfer-prediction version: it asks whether rules learned on other EEG tasks generalize to a new dataset.",
            "",
            "## Feature Counts",
            "",
            "| Feature set | Epoch | N features |",
            "| --- | ---: | ---: |",
        ]
    )
    for (feature_set, epoch), count in sorted(feature_counts.items()):
        lines.append(f"| `{feature_set}` | {epoch} | {count} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_joined(Path(args.joined), args.mode)
    features, targets = aggregate_layer_rows(rows)
    epochs = [int(v) for v in args.epochs.split()]
    protocols = args.protocols.split()
    feature_sets = ("hsdd", "cka_rsa", "early_val", "early_val_hsdd", "all")

    decisions: list[dict[str, object]] = []
    feature_counts: dict[tuple[str, int], int] = {}
    for protocol in protocols:
        for epoch in epochs:
            for feature_set in feature_sets:
                selected, names = evaluate_selector(
                    features=features,
                    targets=targets,
                    epoch=epoch,
                    kind=feature_set,
                    protocol=protocol,
                    ridge=args.ridge,
                )
                decisions.extend(selected)
                feature_counts[(feature_set, epoch)] = len(names)

    summary = summarize(decisions)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "model_selection_decisions.csv", decisions)
    write_csv(out_dir / "model_selection_summary.csv", summary)
    write_markdown(out_dir / "hsdd_guided_model_selection_summary.md", summary, feature_counts)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
