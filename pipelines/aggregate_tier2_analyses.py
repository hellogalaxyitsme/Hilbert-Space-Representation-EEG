#!/usr/bin/env python3
"""Aggregate Tier-2 matched-null and band-power probe analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-dir", default="reports/tier2_matched_nulls")
    parser.add_argument("--probe-dir", default="reports/tier2_bandpower_probes")
    parser.add_argument("--out-dir", default="reports/tier2_stats")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def fmt(value: Any, digits: int = 3) -> str:
    x = finite_float(value)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def flatten_null_rows(null_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(null_dir.glob("*_matched_nulls.json")):
        if path.name.startswith("smoke_"):
            continue
        payload = read_json(path)
        for row in payload.get("rows", []):
            rows.append({**row, "source": str(path)})
    return rows


def flatten_probe_rows(probe_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(probe_dir.glob("*_bandpower_probe.json")):
        if path.name.startswith("smoke_"):
            continue
        payload = read_json(path)
        for row in payload.get("rows", []):
            rows.append({**row, "source": str(path)})
    return rows


def summarize_nulls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["arch"], row["mode"], row["band_scheme"], row["layer"])].append(row)
    out = []
    for (dataset, arch, mode, scheme, layer), vals in sorted(grouped.items()):
        delta = np.asarray([finite_float(v["delta_odi_vs_matched_null"]) for v in vals], dtype=float)
        norm = np.asarray([finite_float(v["odi_over_expected_random"]) for v in vals], dtype=float)
        actual = np.asarray([finite_float(v["actual_odi_mean"]) for v in vals], dtype=float)
        null = np.asarray([finite_float(v["matched_null_odi_mean"]) for v in vals], dtype=float)
        keep = np.isfinite(delta)
        if not np.any(keep):
            continue
        out.append(
            {
                "dataset": dataset,
                "arch": arch,
                "mode": mode,
                "band_scheme": scheme,
                "layer": layer,
                "n_rows": int(np.sum(keep)),
                "actual_odi_mean": float(np.nanmean(actual)),
                "matched_null_odi_mean": float(np.nanmean(null)),
                "delta_odi_mean": float(np.nanmean(delta)),
                "delta_odi_std": float(np.nanstd(delta, ddof=1)) if np.sum(keep) > 1 else 0.0,
                "delta_positive_fraction": float(np.nanmean(delta > 0.0)),
                "odi_over_expected_random_mean": float(np.nanmean(norm)),
            }
        )
    return out


def taxonomy_delta_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    taxonomy = {
        "eegnet": "progressive collapse",
        "shallowconvnet": "nonlinear-induced distortion",
        "eegconformer": "architecture-induced collapse with training reorganization",
        "tsception": "multi-branch fusion-induced reorganization",
        "atcnet": "hybrid convolutional collapse with attention/TCN reorganization",
    }
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("mode") != "true" or row.get("band_scheme") != "canonical":
            continue
        grouped[(row["dataset"], row["arch"], taxonomy.get(row["arch"], "unknown"))].append(
            finite_float(row["delta_odi_vs_matched_null"])
        )
    out = []
    for (dataset, arch, taxon), vals in sorted(grouped.items()):
        arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
        if len(arr) == 0:
            continue
        out.append(
            {
                "dataset": dataset,
                "arch": arch,
                "taxonomy": taxon,
                "n_layer_checkpoint_rows": len(arr),
                "delta_odi_mean": float(arr.mean()),
                "delta_odi_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "delta_positive_fraction": float(np.mean(arr > 0.0)),
            }
        )
    return out


def join_probe_null(null_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    null_by_key = {}
    for row in null_rows:
        key = (
            row["dataset"],
            row["arch"],
            row["mode"],
            int(row["seed"]),
            int(row["epoch"]),
            row["band_scheme"],
            row["layer"],
        )
        null_by_key[key] = row
    joined = []
    for row in probe_rows:
        key = (
            row["dataset"],
            row["arch"],
            row["mode"],
            int(row["seed"]),
            int(row["epoch"]),
            row["band_scheme"],
            row["layer"],
        )
        null = null_by_key.get(key)
        if not null:
            continue
        joined.append(
            {
                "dataset": row["dataset"],
                "arch": row["arch"],
                "mode": row["mode"],
                "seed": row["seed"],
                "epoch": row["epoch"],
                "band_scheme": row["band_scheme"],
                "layer": row["layer"],
                "bandpower_probe_val_acc": finite_float(row["bandpower_probe_val_acc"]),
                "bandpower_probe_train_acc": finite_float(row["bandpower_probe_train_acc"]),
                "actual_odi_mean": finite_float(null["actual_odi_mean"]),
                "delta_odi_vs_matched_null": finite_float(null["delta_odi_vs_matched_null"]),
                "odi_over_expected_random": finite_float(null["odi_over_expected_random"]),
            }
        )
    return joined


def redundancy_correlations(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[(row["dataset"], row["arch"], row["mode"], row["band_scheme"])].append(row)
    out = []
    for (dataset, arch, mode, scheme), vals in sorted(grouped.items()):
        if len(vals) < 4:
            continue
        y = np.asarray([finite_float(v["bandpower_probe_val_acc"]) for v in vals], dtype=float)
        for metric in ("actual_odi_mean", "delta_odi_vs_matched_null", "odi_over_expected_random"):
            x = np.asarray([finite_float(v[metric]) for v in vals], dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            if np.sum(keep) < 4:
                continue
            pear = stats.pearsonr(x[keep], y[keep])
            spear = stats.spearmanr(x[keep], y[keep])
            out.append(
                {
                    "dataset": dataset,
                    "arch": arch,
                    "mode": mode,
                    "band_scheme": scheme,
                    "metric": metric,
                    "n_rows": int(np.sum(keep)),
                    "pearson_r": float(pear.statistic),
                    "pearson_p": float(pear.pvalue),
                    "spearman_r": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                }
            )
    return out


def write_summary(path: Path, null_summary: list[dict[str, Any]], tax_summary: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tier-2 Matched-Null and Band-Power Probe Summary",
        "",
        "This report aggregates dimension-matched null ODI runs and per-layer band-power linear probes.",
        "",
        "## Delta-ODI Taxonomy Summary",
        "",
        "| Dataset | Architecture | Taxonomy | n | Mean delta ODI | Positive fraction |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in tax_summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['arch']}` | {row['taxonomy']} | "
            f"{row['n_layer_checkpoint_rows']} | {fmt(row['delta_odi_mean'])} | {fmt(row['delta_positive_fraction'])} |"
        )
    lines += [
        "",
        "## ODI/Band-Decodability Redundancy Check",
        "",
        "| Dataset | Architecture | Metric | n | Pearson r | Spearman r |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in corr_rows:
        if row["mode"] != "true" or row["band_scheme"] != "canonical":
            continue
        lines.append(
            f"| `{row['dataset']}` | `{row['arch']}` | `{row['metric']}` | "
            f"{row['n_rows']} | {fmt(row['pearson_r'])} | {fmt(row['spearman_r'])} |"
        )
    lines += [
        "",
        "Interpretation rule: ODI is non-redundant with band decodability when delta-ODI and band-power probe accuracy are not tightly coupled, or when their signs differ across architecture families.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    null_rows = flatten_null_rows(Path(args.null_dir))
    probe_rows = flatten_probe_rows(Path(args.probe_dir))
    null_summary = summarize_nulls(null_rows)
    tax_summary = taxonomy_delta_summary(null_rows)
    joined = join_probe_null(null_rows, probe_rows)
    corr_rows = redundancy_correlations(joined)
    write_csv(out_dir / "matched_null_layer_rows.csv", null_rows)
    write_csv(out_dir / "matched_null_layer_summary.csv", null_summary)
    write_csv(out_dir / "delta_odi_taxonomy_summary.csv", tax_summary)
    write_csv(out_dir / "bandpower_probe_rows.csv", probe_rows)
    write_csv(out_dir / "bandpower_probe_null_joined.csv", joined)
    write_csv(out_dir / "odi_bandpower_redundancy_correlations.csv", corr_rows)
    write_summary(out_dir / "tier2_summary.md", null_summary, tax_summary, corr_rows)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
