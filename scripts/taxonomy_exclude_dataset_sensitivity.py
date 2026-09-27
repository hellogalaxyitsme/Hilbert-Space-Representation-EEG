#!/usr/bin/env python3
"""Refit strict leave-dataset-out taxonomy predictions from compact trajectories."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


OUTPUT_FIELDS = (
    "evaluation_scope", "split_id", "train_scope", "heldout_scope", "predictor",
    "taxonomy", "operation_group", "dataset", "arch", "seed", "layer",
    "predicted_sign", "actual_sign", "correct", "actual_pearson_odi_val_acc",
)
PAIR_FIELDS = (
    "evaluation_scope", "split_id", "train_scope", "heldout_scope", "dataset",
    "arch", "seed", "layer", "operation_group",
)
JOINED_REQUIRED = {"dataset", "arch", "mode", "seed", "layer", "epoch", "hsdd_odi", "val_acc"}
TAXONOMY_REQUIRED = {"dataset", "arch", "mode", "seed", "layer", "operation_group", "taxonomy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, required=True, help="Joined layer/checkpoint metric CSV.")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Layer-to-taxonomy mapping CSV.")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset inventory used for fit and evaluation.")
    parser.add_argument("--out", type=Path, required=True, help="Prediction-row output CSV.")
    parser.add_argument("--summary", type=Path, required=True, help="Bootstrap-summary output CSV.")
    parser.add_argument("--fold-table", type=Path, required=True, help="Fold metadata output CSV.")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def validate_draws(draws: int) -> None:
    if draws < 1:
        raise ValueError("--draws must be positive")


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def finite_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def integer(value: str, label: str) -> int:
    parsed = finite_float(value, label)
    if not parsed.is_integer():
        raise ValueError(f"non-integral {label}: {value!r}")
    return int(parsed)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pearson(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def majority(signs: list[int]) -> int | None:
    if not signs:
        return None
    positive = sum(sign > 0 for sign in signs)
    negative = sum(sign < 0 for sign in signs)
    return 1 if positive >= negative else -1


def taxonomy_predictions(trajectories: list[dict[str, object]]) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in trajectories:
        grouped[(str(row["taxonomy"]), str(row["operation_group"]))].append(float(row["pearson"]))
    output = {}
    for key, values in grouped.items():
        mean_value = float(np.mean(values))
        if len(values) >= 3 and abs(mean_value) > 1e-12:
            output[key] = 1 if mean_value > 0 else -1
    return output


def build_trajectories(
    joined_rows: list[dict[str, str]],
    taxonomy_rows: list[dict[str, str]],
    datasets: tuple[str, ...],
) -> tuple[list[dict[str, object]], set[str]]:
    inventory = {row["dataset"] for row in joined_rows if row["mode"] == "true"}
    unknown = set(datasets) - inventory
    if unknown:
        raise ValueError(f"datasets absent from joined records: {sorted(unknown)}")

    mapping: dict[tuple[str, str, str, int, str], tuple[str, str]] = {}
    for row in taxonomy_rows:
        key = (row["dataset"], row["arch"], row["mode"], integer(row["seed"], "taxonomy seed"), row["layer"])
        if key in mapping:
            raise ValueError(f"duplicate taxonomy mapping key: {key}")
        operation_group = row["operation_group"].strip()
        taxonomy = row["taxonomy"].strip()
        if not operation_group or not taxonomy or operation_group == "unmapped" or taxonomy == "unmapped":
            raise ValueError(f"unmapped taxonomy key: {key}")
        mapping[key] = (operation_group, taxonomy)

    grouped: dict[tuple[str, str, int, str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    seen_joined = set()
    for row in joined_rows:
        if row["dataset"] not in datasets or row["mode"] != "true":
            continue
        if row["dataset"] == "p300" and row["arch"] == "atcnet":
            continue
        seed = integer(row["seed"], "joined seed")
        epoch = integer(row["epoch"], "joined epoch")
        source_key = (row["dataset"], row["arch"], row["mode"], seed, row["layer"], epoch)
        if source_key in seen_joined:
            raise ValueError(f"duplicate joined layer/checkpoint key: {source_key}")
        seen_joined.add(source_key)
        mapping_key = source_key[:-1]
        if mapping_key not in mapping:
            raise ValueError(f"unmapped taxonomy key: {mapping_key}")
        operation_group, taxonomy = mapping[mapping_key]
        grouped[(row["dataset"], row["arch"], seed, row["layer"], operation_group, taxonomy)].append(
            (epoch, finite_float(row["hsdd_odi"], "hsdd_odi"), finite_float(row["val_acc"], "val_acc"))
        )

    trajectories = []
    for (dataset, arch, seed, layer, operation_group, taxonomy), points in grouped.items():
        points.sort()
        correlation = pearson([point[1] for point in points], [point[2] for point in points])
        if math.isfinite(correlation) and abs(correlation) > 1e-12:
            trajectories.append({
                "dataset": dataset, "arch": arch, "seed": seed, "layer": layer,
                "operation_group": operation_group, "taxonomy": taxonomy,
                "pearson": correlation, "sign": 1 if correlation > 0 else -1,
                "n_checkpoints": len(points),
            })
    if not trajectories:
        raise ValueError("no eligible non-constant trajectories")
    return trajectories, inventory


def strict_bootstrap(
    predictions: list[dict[str, object]], datasets: tuple[str, ...], draws: int, seed: int
) -> list[dict[str, object]]:
    by_predictor: dict[str, dict[tuple[object, ...], dict[str, object]]] = defaultdict(dict)
    for row in predictions:
        key = tuple(row[field] for field in PAIR_FIELDS)
        predictor = str(row["predictor"])
        if key in by_predictor[predictor]:
            raise ValueError(f"duplicate prediction key: {key}")
        by_predictor[predictor][key] = row
    if set(by_predictor["taxonomy"]) != set(by_predictor["majority_class"]):
        raise ValueError("taxonomy and majority-class predictor coverage differs")
    if not by_predictor["taxonomy"]:
        raise ValueError("no paired taxonomy/majority predictions")

    clusters: dict[str, dict[str, dict[tuple[str, int], list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    paired: dict[str, dict[tuple[str, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    for predictor, rows in by_predictor.items():
        for row in rows.values():
            clusters[str(row["dataset"])][predictor][(str(row["arch"]), int(row["seed"]))].append(int(row["correct"]))
    for key in sorted(by_predictor["taxonomy"]):
        taxonomy = by_predictor["taxonomy"][key]
        majority_row = by_predictor["majority_class"][key]
        paired[str(taxonomy["dataset"])][(str(taxonomy["arch"]), int(taxonomy["seed"]))].append(
            int(taxonomy["correct"]) - int(majority_row["correct"])
        )

    rng = random.Random(seed)
    predictors = sorted(by_predictor)
    for dataset in sorted(datasets):
        for predictor in predictors:
            means = [sum(values) / len(values) for values in clusters[dataset][predictor].values()]
            if not means:
                raise ValueError(f"empty {predictor} cluster set for {dataset}")
            for _ in range(draws):
                _ = sum(rng.choice(means) for _ in means) / len(means)

    output = []
    for predictor in predictors:
        values = [[sum(rows) / len(rows) for rows in clusters[dataset][predictor].values()] for dataset in datasets]
        if any(not value for value in values):
            raise ValueError(f"empty {predictor} cluster set")
        estimate = sum(sum(value) / len(value) for value in values) / len(values)
        draws_values = [
            sum(sum(rng.choice(value) for _ in value) / len(value) for value in values) / len(values)
            for _ in range(draws)
        ]
        if predictor in {"taxonomy", "majority_class"}:
            output.append({
                "scope": f"{len(datasets)}_dataset_mean", "dataset": "+".join(datasets),
                "predictor": predictor, "n_rows": sum(len(rows) for dataset in datasets for rows in clusters[dataset][predictor].values()),
                "n_run_clusters": sum(len(clusters[dataset][predictor]) for dataset in datasets),
                "estimate": estimate, "ci_low": quantile(draws_values, 0.025), "ci_high": quantile(draws_values, 0.975),
                "draws": draws, "seed": seed,
            })
    paired_values = [[sum(rows) / len(rows) for rows in paired[dataset].values()] for dataset in sorted(datasets)]
    if any(not value for value in paired_values):
        raise ValueError("empty paired run-cluster set")
    estimate = sum(sum(value) / len(value) for value in paired_values) / len(paired_values)
    draws_values = [
        sum(sum(rng.choice(value) for _ in value) / len(value) for value in paired_values) / len(paired_values)
        for _ in range(draws)
    ]
    output.append({
        "scope": f"{len(datasets)}_dataset_paired_difference", "dataset": "+".join(datasets),
        "predictor": "taxonomy_minus_majority_class", "n_rows": len(by_predictor["taxonomy"]),
        "n_run_clusters": sum(len(paired[dataset]) for dataset in datasets), "estimate": estimate,
        "ci_low": quantile(draws_values, 0.025), "ci_high": quantile(draws_values, 0.975), "draws": draws, "seed": seed,
    })
    return output


def main() -> None:
    args = parse_args()
    validate_draws(args.draws)
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if len(datasets) < 3 or len(set(datasets)) != len(datasets):
        raise ValueError("--datasets requires at least three unique names")
    trajectories, inventory = build_trajectories(
        read_rows(args.joined, JOINED_REQUIRED), read_rows(args.taxonomy, TAXONOMY_REQUIRED), datasets
    )
    predictions: list[dict[str, object]] = []
    folds = []
    for held_out in datasets:
        training = [row for row in trajectories if row["dataset"] != held_out]
        testing = [row for row in trajectories if row["dataset"] == held_out]
        if not training or not testing:
            raise ValueError(f"empty training or held-out trajectory set for {held_out}")
        taxonomy = taxonomy_predictions(training)
        global_majority = majority([int(row["sign"]) for row in training])
        if global_majority is None:
            raise ValueError(f"empty majority sign set for {held_out}")
        train_scope = "+".join(dataset for dataset in datasets if dataset != held_out)
        folds.append({
            "evaluation_scope": "leave_dataset_out", "heldout_dataset": held_out, "fit_datasets": train_scope,
            "n_train_trajectories": len(training), "n_test_trajectories": len(testing),
            "n_taxonomy_keys_fitted": len(taxonomy), "training_only_global_majority": global_majority,
            "excluded_dataset": "+".join(sorted(inventory - set(datasets))),
        })
        for row in testing:
            for predictor, prediction in (
                ("taxonomy", taxonomy.get((str(row["taxonomy"]), str(row["operation_group"])))),
                ("majority_class", global_majority),
                ("per_dataset_majority", global_majority),
            ):
                if prediction is None:
                    continue
                predictions.append({
                    "evaluation_scope": "leave_dataset_out", "split_id": f"heldout_dataset={held_out}",
                    "train_scope": train_scope, "heldout_scope": held_out, "predictor": predictor,
                    "taxonomy": row["taxonomy"], "operation_group": row["operation_group"], "dataset": row["dataset"],
                    "arch": row["arch"], "seed": row["seed"], "layer": row["layer"],
                    "predicted_sign": prediction, "actual_sign": row["sign"], "correct": int(prediction == row["sign"]),
                    "actual_pearson_odi_val_acc": row["pearson"],
                })

    summary = strict_bootstrap(predictions, datasets, args.draws, args.seed)
    for path, rows, fields in (
        (args.out, predictions, OUTPUT_FIELDS),
        (args.summary, summary, tuple(summary[0])),
        (args.fold_table, folds, tuple(folds[0])),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print({"datasets": datasets, "trajectory_rows": len(trajectories), "prediction_rows": len(predictions), "summary": summary})


if __name__ == "__main__":
    main()
