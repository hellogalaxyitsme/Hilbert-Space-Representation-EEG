#!/usr/bin/env python3
"""Build task-specific chance and balance calibration checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DATASET_REPRESENTATIVES = {
    "bci2a": Path("reports/eegnet_bci2a_layer_audit.json"),
    "sleepedf_full": Path("reports/sleepedf_full_eegnet_layer_audit_true_seed41.json"),
    "seediv": Path("reports/seediv_eegnet_layer_audit_true_seed41.json"),
    "p300": Path("reports/p300_eegnet_layer_audit_true_seed41.json"),
}


DATASET_NOTES = {
    "bci2a": "Four balanced motor-imagery classes; A09T is the original held-out subject/session.",
    "sleepedf_full": "Training/validation use class-balanced sampled epochs from the full Sleep-EDF cache.",
    "seediv": "Subject-heldout validation with balanced sampled emotion classes.",
    "p300": "Raw task is target/non-target imbalanced, but training/validation sampling is balanced.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/calibration_checks")
    parser.add_argument("--multiseed-summary", default="reports/multiseed_stats/mean_std_by_metric.csv")
    parser.add_argument("--fm-summary", default="reports/fm_multiseed_stats/fm_mean_std_by_metric.csv")
    parser.add_argument(
        "--classical-summary",
        default="reports/classical_filterbank_dl_comparison/classical_filterbank_dl_accuracy_summary.csv",
    )
    parser.add_argument(
        "--fbcsp-summary",
        default="reports/bci2a_fbcsp_dl_comparison/bci2a_fbcsp_dl_accuracy_summary.csv",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


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


def counts_to_sorted_dict(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))}


def majority_rate(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    return max(counts.values()) / total


def fmt_counts(counts: dict[str, int]) -> str:
    return "; ".join(f"{k}:{v}" for k, v in counts.items())


def dataset_calibration_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    calibration: dict[str, dict[str, float]] = {}
    for dataset, path in DATASET_REPRESENTATIVES.items():
        payload = read_json(path)
        meta = payload.get("dataset", {})
        train_counts = counts_to_sorted_dict(meta.get("train_label_counts"))
        val_counts = counts_to_sorted_dict(meta.get("val_label_counts"))
        raw_counts = counts_to_sorted_dict(meta.get("label_counts") or meta.get("labels_all"))
        if dataset == "bci2a":
            raw_counts = {k: v for k, v in raw_counts.items() if k != "783"}
        n_classes = len(val_counts) if val_counts else len(train_counts)
        chance = 1.0 / n_classes if n_classes else float("nan")
        val_majority = majority_rate(val_counts)
        train_majority = majority_rate(train_counts)
        raw_majority = majority_rate(raw_counts)
        calibration[dataset] = {
            "chance_accuracy": chance,
            "validation_majority_baseline": val_majority,
            "train_majority_baseline": train_majority,
            "raw_label_majority_baseline": raw_majority,
        }
        rows.append(
            {
                "dataset": dataset,
                "n_classes": n_classes,
                "chance_accuracy": chance,
                "validation_majority_baseline": val_majority,
                "train_majority_baseline": train_majority,
                "raw_label_majority_baseline": raw_majority,
                "validation_is_balanced": abs(val_majority - chance) < 1e-12
                if math.isfinite(val_majority) and math.isfinite(chance)
                else False,
                "train_label_counts": fmt_counts(train_counts),
                "val_label_counts": fmt_counts(val_counts),
                "raw_label_counts": fmt_counts(raw_counts),
                "note": DATASET_NOTES[dataset],
                "source": str(path),
            }
        )
    return rows, calibration


def find_summary(rows: list[dict[str, str]], **query: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in query.items()):
            return row
    return None


def add_performance_row(
    rows: list[dict[str, Any]],
    calibration: dict[str, dict[str, float]],
    *,
    dataset: str,
    family: str,
    model: str,
    experiment: str,
    true_mean: float,
    true_std: float,
    shuffled_mean: float = float("nan"),
    untrained_mean: float = float("nan"),
) -> None:
    chance = calibration[dataset]["chance_accuracy"]
    majority = calibration[dataset]["validation_majority_baseline"]
    rows.append(
        {
            "dataset": dataset,
            "family": family,
            "model": model,
            "experiment": experiment,
            "chance_accuracy": chance,
            "validation_majority_baseline": majority,
            "true_mean": true_mean,
            "true_std": true_std,
            "true_minus_chance": true_mean - chance,
            "true_minus_validation_majority": true_mean - majority,
            "shuffled_mean": shuffled_mean,
            "true_minus_shuffled": true_mean - shuffled_mean if math.isfinite(shuffled_mean) else float("nan"),
            "untrained_mean": untrained_mean,
            "true_minus_untrained": true_mean - untrained_mean if math.isfinite(untrained_mean) else float("nan"),
        }
    )


def performance_rows(args: argparse.Namespace, calibration: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    multiseed = read_csv(Path(args.multiseed_summary))
    fm = read_csv(Path(args.fm_summary))
    classical = read_csv(Path(args.classical_summary))
    fbcsp = read_csv(Path(args.fbcsp_summary))

    arches = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
    datasets = ("bci2a", "sleepedf_full", "seediv", "p300")
    for dataset in datasets:
        for arch in arches:
            true = find_summary(
                multiseed,
                dataset=dataset,
                arch=arch,
                experiment="static",
                condition="true",
                metric="training.best_val_acc",
            )
            if not true:
                continue
            shuffled = find_summary(
                multiseed,
                dataset=dataset,
                arch=arch,
                experiment="static",
                condition="shuffled",
                metric="training.best_val_acc",
            )
            untrained = find_summary(
                multiseed,
                dataset=dataset,
                arch=arch,
                experiment="static",
                condition="untrained",
                metric="training.best_val_acc",
            )
            add_performance_row(
                out,
                calibration,
                dataset=dataset,
                family="supervised_dl",
                model=arch,
                experiment="static_best_val_acc",
                true_mean=finite_float(true["mean"]),
                true_std=finite_float(true["std"]),
                shuffled_mean=finite_float(shuffled["mean"]) if shuffled else float("nan"),
                untrained_mean=finite_float(untrained["mean"]) if untrained else float("nan"),
            )

    fms = ("biot", "labram", "reve", "eegpt")
    for dataset in datasets:
        for model in fms:
            true = find_summary(fm, dataset=dataset, fm=model, mode="true", metric="best_val_acc")
            if not true:
                continue
            shuffled = find_summary(fm, dataset=dataset, fm=model, mode="shuffled", metric="best_val_acc")
            untrained = find_summary(fm, dataset=dataset, fm=model, mode="untrained", metric="best_val_acc")
            add_performance_row(
                out,
                calibration,
                dataset=dataset,
                family="frozen_fm",
                model=model,
                experiment="linear_probe_best_val_acc",
                true_mean=finite_float(true["mean"]),
                true_std=finite_float(true["std"]),
                shuffled_mean=finite_float(shuffled["mean"]) if shuffled else float("nan"),
                untrained_mean=finite_float(untrained["mean"]) if untrained else float("nan"),
            )

    for row in fbcsp:
        if row.get("family") == "fbcsp" and row.get("model") == "fbcsp" and row.get("mode") == "true":
            shuffled = find_summary(fbcsp, family="fbcsp", model="fbcsp", mode="shuffled")
            add_performance_row(
                out,
                calibration,
                dataset="bci2a",
                family="classical_eeg",
                model="fbcsp",
                experiment="best_val_acc",
                true_mean=finite_float(row["val_acc_mean"]),
                true_std=finite_float(row["val_acc_std"]),
                shuffled_mean=finite_float(shuffled["val_acc_mean"]) if shuffled else float("nan"),
            )
    for row in classical:
        if row.get("family") == "classical_filterbank" and row.get("mode") == "true":
            dataset = row["dataset"]
            shuffled = find_summary(
                classical,
                dataset=dataset,
                family="classical_filterbank",
                model=row["model"],
                mode="shuffled",
            )
            add_performance_row(
                out,
                calibration,
                dataset=dataset,
                family="classical_eeg",
                model=row["model"],
                experiment="best_val_acc",
                true_mean=finite_float(row["val_acc_mean"]),
                true_std=finite_float(row["val_acc_std"]),
                shuffled_mean=finite_float(shuffled["val_acc_mean"]) if shuffled else float("nan"),
            )
    return sorted(out, key=lambda r: (r["dataset"], r["family"], r["model"], r["experiment"]))


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.3f}"


def write_summary(out_dir: Path, dataset_rows: list[dict[str, Any]], perf_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Task Calibration and Chance/Balance Checks",
        "",
        "This report calibrates reported accuracies against task-specific chance and validation-set majority baselines.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'dataset_calibration_summary.csv'}`",
        f"- `{out_dir / 'performance_margin_summary.csv'}`",
        "",
        "## Dataset Calibration",
        "",
        "| Dataset | Classes | Chance | Validation majority | Validation balanced | Note |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in dataset_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['n_classes']} | {fmt(row['chance_accuracy'])} | "
            f"{fmt(row['validation_majority_baseline'])} | {row['validation_is_balanced']} | {row['note']} |"
        )
    lines += [
        "",
        "## Primary Accuracy Margins",
        "",
        "| Dataset | Family | Model | True acc | Chance margin | Majority margin | True-shuffled | True-untrained |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in perf_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['family']} | `{row['model']}` | "
            f"{fmt(row['true_mean'])} +/- {fmt(row['true_std'])} | "
            f"{fmt(row['true_minus_chance'])} | {fmt(row['true_minus_validation_majority'])} | "
            f"{fmt(row['true_minus_shuffled'])} | {fmt(row['true_minus_untrained'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- BCI IV 2a, Sleep-EDF full, and SEED-IV validation sets are balanced in the report metadata, so chance equals the validation majority baseline.",
        "- P300 uses a balanced validation protocol, so the validation metric is balanced P300 accuracy. Raw event-majority accuracy describes the unbalanced event distribution.",
        "- Shuffled-label and untrained-head/control gaps should be reported alongside chance margins to avoid overstating small above-chance differences.",
    ]
    (out_dir / "calibration_checks_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    dataset_rows, calibration = dataset_calibration_rows()
    perf_rows = performance_rows(args, calibration)
    write_csv(out_dir / "dataset_calibration_summary.csv", dataset_rows)
    write_csv(out_dir / "performance_margin_summary.csv", perf_rows)
    write_summary(out_dir, dataset_rows, perf_rows)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
