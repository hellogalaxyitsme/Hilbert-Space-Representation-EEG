#!/usr/bin/env python3
"""Aggregate matched HSDD, CKA, and RSA dynamics metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from hsdd.diagnostics import _spearman
from hsdd.io import write_json


DATASETS = ("bci2a", "sleepedf_full", "seediv", "p300")
ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
MODES = ("true", "shuffled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cka-dir", default="reports/cka_rsa")
    parser.add_argument("--out-dir", default="reports/cka_rsa_stats")
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--arches", default=" ".join(ARCHES))
    parser.add_argument("--modes", default=" ".join(MODES))
    parser.add_argument("--seeds", default="41 42 43 44 45 46")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def no_suffix_seed41_path(path: Path, seed: int) -> Path:
    if seed == 41:
        alt = Path(str(path).replace("_seed41", ""))
        if alt.exists():
            return alt
    return path


def dynamics_path(reports_dir: Path, dataset: str, arch: str, mode: str, seed: int) -> Path:
    if dataset == "bci2a":
        path = reports_dir / f"{arch}_bci2a_dynamics_{mode}_seed{seed}.json"
    else:
        path = reports_dir / f"{dataset}_{arch}_dynamics_{mode}_seed{seed}.json"
    return no_suffix_seed41_path(path, seed)


def cka_path(cka_dir: Path, dataset: str, arch: str, mode: str, seed: int) -> Path:
    return cka_dir / f"{dataset}_{arch}_dynamics_{mode}_seed{seed}_cka_rsa.json"


def pearson(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / den)


def fmt(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def flatten_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    reports_dir = Path(args.reports_dir)
    cka_dir = Path(args.cka_dir)
    datasets = args.datasets.split()
    arches = args.arches.split()
    modes = args.modes.split()
    seeds = [int(v) for v in args.seeds.split()]
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for arch in arches:
            for mode in modes:
                for seed in seeds:
                    cka_file = cka_path(cka_dir, dataset, arch, mode, seed)
                    dyn_file = dynamics_path(reports_dir, dataset, arch, mode, seed)
                    if not cka_file.exists() or not dyn_file.exists():
                        continue
                    cka = read_json(cka_file)
                    dyn = read_json(dyn_file)
                    dyn_by_epoch = {int(row["epoch"]): row for row in dyn.get("audits", [])}
                    for audit in cka.get("audits", []):
                        epoch = int(audit["epoch"])
                        dyn_audit = dyn_by_epoch.get(epoch, {})
                        for layer, cka_metrics in audit.get("layers", {}).items():
                            hsdd_layer = dyn_audit.get("layers", {}).get(layer, {})
                            pair = hsdd_layer.get("pairwise_isometry", {})
                            band = hsdd_layer.get("band_orthogonality", {})
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "arch": arch,
                                    "mode": mode,
                                    "seed": seed,
                                    "epoch": epoch,
                                    "layer": layer,
                                    "train_acc": audit.get("train_acc"),
                                    "val_acc": audit.get("val_acc"),
                                    "hsdd_distance_spearman": pair.get("distance_spearman"),
                                    "hsdd_scaled_ratio_error_mean": pair.get("scaled_abs_ratio_error_mean"),
                                    "hsdd_odi": band.get("output_mean_abs_offdiag_cosine_mean"),
                                    "hsdd_collapse_factor": band.get("collapse_factor_mean"),
                                    "cka_input": cka_metrics.get("linear_cka_input"),
                                    "rsa_input_spearman": cka_metrics.get("rsa_input_spearman"),
                                    "rsa_label_spearman": cka_metrics.get("rsa_label_spearman"),
                                    "activation_rank": cka_metrics.get("activation_rank"),
                                }
                            )
    return rows


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


def trajectory_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["arch"], row["mode"], int(row["seed"]), row["layer"])
        grouped.setdefault(key, []).append(row)

    metrics = (
        "hsdd_odi",
        "hsdd_distance_spearman",
        "hsdd_scaled_ratio_error_mean",
        "cka_input",
        "rsa_input_spearman",
        "rsa_label_spearman",
    )
    out = []
    for (dataset, arch, mode, seed, layer), vals in grouped.items():
        vals = sorted(vals, key=lambda r: int(r["epoch"]))
        val_acc = [float(v["val_acc"]) for v in vals if v.get("val_acc") is not None]
        if len(val_acc) < 2:
            continue
        for metric in metrics:
            xs = []
            ys = []
            for v in vals:
                if v.get("val_acc") is None or v.get(metric) is None:
                    continue
                try:
                    xs.append(float(v[metric]))
                    ys.append(float(v["val_acc"]))
                except (TypeError, ValueError):
                    continue
            if len(xs) < 2:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "arch": arch,
                    "mode": mode,
                    "seed": seed,
                    "layer": layer,
                    "metric": metric,
                    "n_checkpoints": len(xs),
                    "pearson_val_acc": pearson(xs, ys),
                    "spearman_val_acc": _spearman(np.asarray(xs), np.asarray(ys)),
                }
            )
    return out


def summarize_correlations(corr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in corr_rows:
        grouped.setdefault((row["dataset"], row["metric"]), []).append(row)
    summary = []
    for (dataset, metric), vals in sorted(grouped.items()):
        pearson_vals = np.asarray([float(v["pearson_val_acc"]) for v in vals], dtype=np.float64)
        spearman_vals = np.asarray([float(v["spearman_val_acc"]) for v in vals], dtype=np.float64)
        pearson_vals = pearson_vals[np.isfinite(pearson_vals)]
        spearman_vals = spearman_vals[np.isfinite(spearman_vals)]
        if len(pearson_vals) == 0:
            continue
        summary.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n_trajectories": int(len(pearson_vals)),
                "mean_pearson_val_acc": float(np.mean(pearson_vals)),
                "std_pearson_val_acc": float(np.std(pearson_vals)),
                "mean_abs_pearson_val_acc": float(np.mean(np.abs(pearson_vals))),
                "positive_fraction_pearson": float(np.mean(pearson_vals > 0.0)),
                "mean_spearman_val_acc": float(np.mean(spearman_vals)) if len(spearman_vals) else float("nan"),
            }
        )
    return summary


def write_markdown(path: Path, rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_reports = len({(r["dataset"], r["arch"], r["mode"], r["seed"]) for r in rows})
    lines = [
        "# CKA/RSA vs HSDD Dynamics Comparison",
        "",
        f"Matched layer-checkpoint rows: {len(rows)}.",
        f"CKA/RSA reports included: {n_reports}.",
        "",
        "Metrics:",
        "",
        "- `cka_input`: linear CKA between raw input epochs and layer activations.",
        "- `rsa_input_spearman`: Spearman correlation between raw-input Euclidean RDM and layer Euclidean RDM.",
        "- `rsa_label_spearman`: Spearman correlation between label-different RDM and layer Euclidean RDM.",
        "- `hsdd_distance_spearman`: HSDD pairwise input-output distance-rank preservation.",
        "- `hsdd_odi`: HSDD band-subspace orthogonality distortion/collapse.",
        "",
        "## Mean Dynamics Correlation With Validation Accuracy",
        "",
        "| Dataset | Metric | Trajectories | Mean Pearson | Mean |r| | Positive frac |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | `{row['metric']}` | {row['n_trajectories']} | "
            f"{fmt(row['mean_pearson_val_acc'])} | {fmt(row['mean_abs_pearson_val_acc'])} | "
            f"{fmt(row['positive_fraction_pearson'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- CKA and input-RSA are best interpreted as global representational similarity to the raw input geometry.",
            "- Label-RSA is complementary: it asks whether layer distances increasingly separate labels, regardless of preserving raw EEG geometry.",
            "- HSDD's pairwise distance Spearman overlaps most directly with input-RSA, but HSDD also reports scale-adjusted metric distortion.",
            "- HSDD's ODI has no direct CKA/RSA equivalent here: it measures whether canonical EEG frequency-band components remain orthogonal or collapse/mix after a layer.",
            "- Evidence for the paper is strongest when label-sensitive validation gains align with HSDD geometry changes while CKA/RSA stay flat or only describe global similarity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = flatten_rows(args)
    corr_rows = trajectory_correlations(rows)
    summary = summarize_correlations(corr_rows)

    write_csv(out_dir / "joined_layer_checkpoint_metrics.csv", rows)
    write_csv(out_dir / "method_val_acc_correlations.csv", corr_rows)
    write_csv(out_dir / "method_val_acc_correlation_summary.csv", summary)
    write_json(out_dir / "method_val_acc_correlation_summary.json", summary)
    write_markdown(out_dir / "cka_rsa_hsdd_summary.md", rows, corr_rows, summary)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
