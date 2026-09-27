#!/usr/bin/env python3
"""Aggregate HSDD hyperparameter sensitivity reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/hsdd_sensitivity")
    parser.add_argument("--out-dir", default="reports/hsdd_sensitivity_stats")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()) if len(arr) else float("nan"), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        grouped.setdefault(tuple(row[k] for k in keys), []).append(float(row[metric]))
    out = []
    for key, vals in sorted(grouped.items()):
        mean, std = mean_std(vals)
        out.append(
            {
                **dict(zip(keys, key)),
                "metric": metric,
                "n": len(vals),
                "mean": mean,
                "std": std,
                "cv_abs": std / max(abs(mean), 1e-12),
                "mean_pm_std": f"{mean:.3f} +/- {std:.3f}",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    paths = sorted(path for path in reports_dir.glob("*_sensitivity.json") if not path.name.startswith("smoke_"))
    if not paths:
        raise FileNotFoundError(f"no sensitivity reports found in {reports_dir}")
    pairwise_rows = []
    band_rows = []
    band_defs = []
    for path in paths:
        payload = read_json(path)
        dataset_name = payload.get("dataset", {}).get("name") or "bci2a"
        for row in payload.get("pairwise_rows", []):
            row.setdefault("dataset", dataset_name)
            pairwise_rows.append(row)
        for row in payload.get("band_rows", []):
            row.setdefault("dataset", dataset_name)
            band_rows.append(row)
        for scheme, bands in payload.get("sensitivity", {}).get("band_schemes", {}).items():
            for band in bands:
                band_defs.append({"source": str(path), "band_scheme": scheme, "band": band[0], "low": band[1], "high": band[2]})

    pairwise_summary = []
    for metric in ("distance_spearman", "scaled_abs_ratio_error_mean"):
        pairwise_summary.extend(
            summarize(pairwise_rows, ("dataset", "arch", "label_mode", "layer", "distance_metric", "fraction"), metric)
        )
    band_summary = []
    for metric in ("output_mean_abs_offdiag_cosine_mean", "collapse_factor_mean"):
        band_summary.extend(summarize(band_rows, ("dataset", "arch", "label_mode", "layer", "band_scheme", "fraction"), metric))

    stability_rows = []
    for rows, family, keys, metrics in [
        (pairwise_rows, "pairwise", ("dataset", "arch", "label_mode", "layer", "distance_metric", "fraction"), ("distance_spearman", "scaled_abs_ratio_error_mean")),
        (band_rows, "band_odi", ("dataset", "arch", "label_mode", "layer", "band_scheme", "fraction"), ("output_mean_abs_offdiag_cosine_mean",)),
    ]:
        for metric in metrics:
            stability_rows.extend(
                {**row, "family": family}
                for row in summarize(rows, keys, metric)
            )

    write_csv(out_dir / "pairwise_sensitivity_rows.csv", pairwise_rows)
    write_csv(out_dir / "band_sensitivity_rows.csv", band_rows)
    write_csv(out_dir / "pairwise_sensitivity_summary.csv", pairwise_summary)
    write_csv(out_dir / "band_sensitivity_summary.csv", band_summary)
    write_csv(out_dir / "sample_size_stability_summary.csv", stability_rows)
    write_csv(out_dir / "band_definitions.csv", band_defs)

    def avg_cv(family: str, metric: str, fraction: float) -> float:
        vals = [
            float(row["cv_abs"])
            for row in stability_rows
            if row["family"] == family and row["metric"] == metric and abs(float(row["fraction"]) - fraction) < 1e-9
        ]
        return float(np.mean(vals)) if vals else float("nan")

    band_scheme_rows = [
        row
        for row in band_summary
        if row["metric"] == "output_mean_abs_offdiag_cosine_mean" and abs(float(row["fraction"]) - 1.0) < 1e-9
    ]
    band_by_scheme: dict[str, list[float]] = {}
    for row in band_scheme_rows:
        band_by_scheme.setdefault(str(row["band_scheme"]), []).append(float(row["mean"]))
    band_by_dataset_scheme: dict[tuple[str, str], list[float]] = {}
    for row in band_scheme_rows:
        band_by_dataset_scheme.setdefault((str(row["dataset"]), str(row["band_scheme"])), []).append(float(row["mean"]))

    distance_rows = [
        row
        for row in pairwise_summary
        if row["metric"] == "distance_spearman" and abs(float(row["fraction"]) - 1.0) < 1e-9
    ]
    dist_by_metric: dict[str, list[float]] = {}
    for row in distance_rows:
        dist_by_metric.setdefault(str(row["distance_metric"]), []).append(float(row["mean"]))
    dist_by_dataset_metric: dict[tuple[str, str], list[float]] = {}
    for row in distance_rows:
        dist_by_dataset_metric.setdefault((str(row["dataset"]), str(row["distance_metric"])), []).append(float(row["mean"]))

    lines = [
        "# HSDD Hyperparameter Sensitivity Summary",
        "",
        f"Reports aggregated: {len(paths)}.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'pairwise_sensitivity_rows.csv'}`",
        f"- `{out_dir / 'band_sensitivity_rows.csv'}`",
        f"- `{out_dir / 'pairwise_sensitivity_summary.csv'}`",
        f"- `{out_dir / 'band_sensitivity_summary.csv'}`",
        f"- `{out_dir / 'sample_size_stability_summary.csv'}`",
        f"- `{out_dir / 'band_definitions.csv'}`",
        "",
        "## Band Definition Sensitivity",
        "",
        "Mean ODI at 100% audit-pool sample, averaged over architectures and compact layers.",
        "",
        "| Band scheme | Mean ODI | Std across layer summaries |",
        "| --- | ---: | ---: |",
    ]
    for scheme, vals in sorted(band_by_scheme.items()):
        mean, std = mean_std(vals)
        lines.append(f"| `{scheme}` | {fmt(mean)} | {fmt(std)} |")
    lines += [
        "",
        "Dataset-specific ODI means at 100% audit-pool sample:",
        "",
        "| Dataset | Band scheme | Mean ODI | Std across layer summaries |",
        "| --- | --- | ---: | ---: |",
    ]
    for (dataset, scheme), vals in sorted(band_by_dataset_scheme.items()):
        mean, std = mean_std(vals)
        lines.append(f"| `{dataset}` | `{scheme}` | {fmt(mean)} | {fmt(std)} |")
    lines += [
        "",
        "## Distance Metric Choice",
        "",
        "Mean pairwise distance Spearman at 100% audit-pool sample, averaged over architectures and compact layers.",
        "",
        "| Distance metric | Mean distance Spearman | Std across layer summaries |",
        "| --- | ---: | ---: |",
    ]
    for metric, vals in sorted(dist_by_metric.items()):
        mean, std = mean_std(vals)
        lines.append(f"| `{metric}` | {fmt(mean)} | {fmt(std)} |")
    lines += [
        "",
        "Dataset-specific distance-Spearman means at 100% audit-pool sample:",
        "",
        "| Dataset | Distance metric | Mean distance Spearman | Std across layer summaries |",
        "| --- | --- | ---: | ---: |",
    ]
    for (dataset, metric), vals in sorted(dist_by_dataset_metric.items()):
        mean, std = mean_std(vals)
        lines.append(f"| `{dataset}` | `{metric}` | {fmt(mean)} | {fmt(std)} |")
    lines += [
        "",
        "## Sample Size Stability",
        "",
        "Mean coefficient of variation across repeated subsamples. Lower is more stable.",
        "",
        "| Fraction | Pairwise Spearman CV | Pairwise scaled-error CV | ODI CV |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for fraction in sorted({float(row["fraction"]) for row in stability_rows}):
        lines.append(
            f"| {fraction:.2f} | "
            f"{fmt(avg_cv('pairwise', 'distance_spearman', fraction))} | "
            f"{fmt(avg_cv('pairwise', 'scaled_abs_ratio_error_mean', fraction))} | "
            f"{fmt(avg_cv('band_odi', 'output_mean_abs_offdiag_cosine_mean', fraction))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This suite changes HSDD computation choices without retraining the models.",
        "- Band-scheme sensitivity reports whether ODI conclusions are tied to canonical clinical bands or persist under narrower and data-driven bands.",
        "- Sample-fraction stability estimates how noisy HSDD metrics are when the audit set is subsampled at 25%, 50%, 75%, and 100%.",
        "- Distance-metric sensitivity checks whether pairwise geometry conclusions depend on Euclidean distance or remain similar under cosine, correlation, and L1 distances.",
    ]
    (out_dir / "hsdd_sensitivity_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
