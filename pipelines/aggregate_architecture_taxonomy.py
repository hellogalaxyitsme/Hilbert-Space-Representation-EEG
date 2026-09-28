#!/usr/bin/env python3
"""Aggregate architecture-dependent HSDD taxonomy evidence across datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DATASETS = ["bci2a", "sleepedf_full", "seediv", "p300"]
ARCHES = ["eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet"]
MODES = ["true", "shuffled"]
SEEDS = [41, 42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out-dir", default="reports/architecture_taxonomy_all_datasets")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def report_candidates(reports_dir: Path, dataset: str, arch: str, mode: str, seed: int) -> list[Path]:
    candidates = [
        reports_dir / f"{dataset}_{arch}_dynamics_{mode}_seed{seed}.json",
        reports_dir / f"{arch}_{dataset}_dynamics_{mode}_seed{seed}.json",
    ]
    if seed == 41:
        candidates.extend(
            [
                reports_dir / f"{dataset}_{arch}_dynamics_{mode}.json",
                reports_dir / f"{arch}_{dataset}_dynamics_{mode}.json",
            ]
        )
    return candidates


def find_report(reports_dir: Path, dataset: str, arch: str, mode: str, seed: int) -> Path | None:
    for path in report_candidates(reports_dir, dataset, arch, mode, seed):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def layer_group(arch: str, layer: str) -> str:
    if arch == "eegnet":
        if "temporal" in layer or "spatial" in layer or layer == "bnorm_1":
            return "early_filtering"
        if "separable" in layer or layer in {"bnorm_2", "pool_2"}:
            return "separable_body"
        if "final_layer" in layer or layer == "logits":
            return "classifier"
    if arch == "shallowconvnet":
        if "conv" in layer or layer == "bnorm":
            return "linear_filtering"
        if "square" in layer:
            return "square_power"
        if "pool" in layer:
            return "pooling"
        if "log" in layer:
            return "log_power"
        if "final_layer" in layer or layer == "logits":
            return "classifier"
    if arch == "eegconformer":
        if "patch_embedding" in layer:
            return "patch_embedding"
        if "transformer" in layer:
            return "transformer"
        if layer in {"fc", "logits"} or "final_layer" in layer:
            return "classifier"
    if arch == "tsception":
        if layer.startswith("temporal_blocks") or "temporal" in layer:
            return "temporal_multiscale"
        if layer.startswith("spatial_block") or "spatial" in layer:
            return "spatial_asymmetry"
        if layer in {"dense_layer", "final_layer", "logits"}:
            return "fusion_classifier"
    if arch == "atcnet":
        if layer == "conv_block":
            return "conv_projection"
        if layer.startswith("attention_blocks"):
            return "attention"
        if layer.startswith("temporal_conv_nets"):
            return "tcn"
        if layer.startswith("final_layer") or layer == "logits":
            return "classifier"
    return "other"


def taxonomy_label(arch: str) -> str:
    return {
        "eegnet": "progressive collapse",
        "shallowconvnet": "nonlinear-induced distortion",
        "eegconformer": "architecture-induced collapse with training reorganization",
        "tsception": "multi-branch fusion-induced reorganization",
        "atcnet": "hybrid convolutional collapse with attention/TCN reorganization",
    }[arch]


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layer_metric(layer_payload: dict[str, Any], family: str, key: str) -> float:
    return float(layer_payload[family][key])


def best_val(report: dict[str, Any]) -> float:
    vals = []
    for row in report.get("history", []):
        if row.get("val_acc") is not None:
            vals.append(float(row["val_acc"]))
    for audit in report.get("audits", []):
        if audit.get("val_acc") is not None:
            vals.append(float(audit["val_acc"]))
    return max(vals) if vals else float("nan")


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for arch in ARCHES:
            for mode in MODES:
                for seed in SEEDS:
                    path = find_report(reports_dir, dataset, arch, mode, seed)
                    if path is None:
                        missing_rows.append({"dataset": dataset, "arch": arch, "mode": mode, "seed": seed})
                        continue
                    report = read_json(path)
                    audits = report.get("audits", [])
                    if not audits:
                        missing_rows.append(
                            {"dataset": dataset, "arch": arch, "mode": mode, "seed": seed, "reason": "no_audits"}
                        )
                        continue
                    first = min(audits, key=lambda row: row["epoch"])
                    last = max(audits, key=lambda row: row["epoch"])
                    final_val = float(last["val_acc"]) if last.get("val_acc") is not None else float("nan")
                    best = best_val(report)
                    for layer, first_payload in first.get("layers", {}).items():
                        if layer not in last.get("layers", {}):
                            continue
                        last_payload = last["layers"][layer]
                        odi0 = layer_metric(
                            first_payload, "band_orthogonality", "output_mean_abs_offdiag_cosine_mean"
                        )
                        odif = layer_metric(last_payload, "band_orthogonality", "output_mean_abs_offdiag_cosine_mean")
                        sp0 = layer_metric(first_payload, "pairwise_isometry", "distance_spearman")
                        spf = layer_metric(last_payload, "pairwise_isometry", "distance_spearman")
                        raw_rows.append(
                            {
                                "dataset": dataset,
                                "arch": arch,
                                "taxonomy": taxonomy_label(arch),
                                "mode": mode,
                                "seed": seed,
                                "layer": layer,
                                "operation_group": layer_group(arch, layer),
                                "epoch0": first["epoch"],
                                "final_epoch": last["epoch"],
                                "odi_epoch0": odi0,
                                "odi_final": odif,
                                "odi_delta": odif - odi0,
                                "spearman_epoch0": sp0,
                                "spearman_final": spf,
                                "spearman_delta": spf - sp0,
                                "best_val_acc": best,
                                "final_val_acc": final_val,
                                "source": str(path),
                            }
                        )

    group_summary: list[dict[str, Any]] = []
    keys = ["dataset", "arch", "taxonomy", "mode", "operation_group"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    for key, rows in sorted(groups.items()):
        out = dict(zip(keys, key))
        for metric in ("odi_epoch0", "odi_final", "odi_delta", "spearman_epoch0", "spearman_final", "spearman_delta"):
            vals = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = std(vals)
        out["n_layer_seed_values"] = len(rows)
        group_summary.append(out)

    arch_summary: list[dict[str, Any]] = []
    arch_keys = ["dataset", "arch", "taxonomy", "mode"]
    arch_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    report_seen: set[tuple[Any, ...]] = set()
    for row in raw_rows:
        arch_groups.setdefault(tuple(row[k] for k in arch_keys), []).append(row)
        report_seen.add((row["dataset"], row["arch"], row["mode"], row["seed"]))
    for key, rows in sorted(arch_groups.items()):
        out = dict(zip(arch_keys, key))
        for metric in ("odi_epoch0", "odi_final", "odi_delta", "spearman_epoch0", "spearman_final", "spearman_delta"):
            vals = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = std(vals)
        report_keys = {(row["dataset"], row["arch"], row["mode"], row["seed"]) for row in rows}
        best_vals = []
        final_vals = []
        for report_key in report_keys:
            rr = next(row for row in rows if (row["dataset"], row["arch"], row["mode"], row["seed"]) == report_key)
            best_vals.append(float(rr["best_val_acc"]))
            final_vals.append(float(rr["final_val_acc"]))
        out["best_val_acc_mean"] = mean(best_vals)
        out["best_val_acc_std"] = std(best_vals)
        out["final_val_acc_mean"] = mean(final_vals)
        out["final_val_acc_std"] = std(final_vals)
        out["n_reports"] = len(report_keys)
        arch_summary.append(out)

    write_csv(out_dir / "architecture_taxonomy_layer_rows.csv", raw_rows)
    write_csv(out_dir / "architecture_taxonomy_operation_summary.csv", group_summary)
    write_csv(out_dir / "architecture_taxonomy_arch_summary.csv", arch_summary)
    write_csv(out_dir / "architecture_taxonomy_missing_reports.csv", missing_rows)

    def operation_row(dataset: str, arch: str, mode: str, group: str) -> dict[str, Any] | None:
        matches = [
            row
            for row in group_summary
            if row["dataset"] == dataset and row["arch"] == arch and row["mode"] == mode and row["operation_group"] == group
        ]
        return matches[0] if matches else None

    def arch_acc(dataset: str, arch: str, mode: str) -> dict[str, Any] | None:
        matches = [row for row in arch_summary if row["dataset"] == dataset and row["arch"] == arch and row["mode"] == mode]
        return matches[0] if matches else None

    lines = [
        "# Architecture Taxonomy Across All Dataset Scopes",
        "",
        "This generated report extends the architecture-dependent HSDD taxonomy to TSception and ATCNet across all reported-scope supervised dataset scopes: BCI IV 2a, Sleep-EDF full, SEED-IV, and P300.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'architecture_taxonomy_layer_rows.csv'}`",
        f"- `{out_dir / 'architecture_taxonomy_operation_summary.csv'}`",
        f"- `{out_dir / 'architecture_taxonomy_arch_summary.csv'}`",
        f"- `{out_dir / 'architecture_taxonomy_missing_reports.csv'}`",
        "",
        f"Missing dynamics reports detected: {len(missing_rows)}.",
        "",
        "## Expanded Taxonomy",
        "",
        "| Taxon | Architectures | Defining operation | HSDD signature |",
        "| --- | --- | --- | --- |",
        "| Progressive collapse | EEGNet | depthwise/separable convolution | early/body ODI starts low and rises with task learning |",
        "| Nonlinear-induced distortion | ShallowConvNet | square/pool/log band-power path | sharp distortion appears before training |",
        "| Architecture-induced collapse with training reorganization | EEGConformer | patch embedding + transformer attention | near-maximal initial collapse followed by partial reorganization |",
        "| Multi-branch fusion-induced reorganization | TSception | multi-scale temporal blocks + asymmetric spatial/fusion layers | branch/fusion layers start high ODI; true-label training usually reduces or reorganizes high-level collapse |",
        "| Hybrid convolutional collapse with attention/TCN reorganization | ATCNet | convolutional projection + attention + temporal convolutional network | conv block progressively collapses from low ODI; attention/TCN layers start high and reorganize |",
        "",
        "## TSception Evidence",
        "",
        "Mean ODI values are averaged over seeds and layers within each operation group.",
        "",
        "| Dataset scope | Operation group | True ODI epoch 0 | True ODI final | True delta | Shuffled delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for group in ("temporal_multiscale", "spatial_asymmetry", "fusion_classifier"):
            true_row = operation_row(dataset, "tsception", "true", group)
            shuf_row = operation_row(dataset, "tsception", "shuffled", group)
            if not true_row:
                continue
            lines.append(
                f"| `{dataset}` | `{group}` | {fmt(true_row['odi_epoch0_mean'])} | {fmt(true_row['odi_final_mean'])} | {fmt(true_row['odi_delta_mean'])} | {fmt(shuf_row['odi_delta_mean'] if shuf_row else float('nan'))} |"
            )
    lines.extend(
        [
            "",
            "## ATCNet Evidence",
            "",
            "| Dataset scope | Operation group | True ODI epoch 0 | True ODI final | True delta | Shuffled delta |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        for group in ("conv_projection", "attention", "tcn", "classifier"):
            true_row = operation_row(dataset, "atcnet", "true", group)
            shuf_row = operation_row(dataset, "atcnet", "shuffled", group)
            if not true_row:
                continue
            lines.append(
                f"| `{dataset}` | `{group}` | {fmt(true_row['odi_epoch0_mean'])} | {fmt(true_row['odi_final_mean'])} | {fmt(true_row['odi_delta_mean'])} | {fmt(shuf_row['odi_delta_mean'] if shuf_row else float('nan'))} |"
            )
    lines.extend(
        [
            "",
            "## Accuracy Context",
            "",
            "| Dataset scope | Architecture | True best val acc | Shuffled best val acc | True-shuffled gap |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        for arch in ("tsception", "atcnet"):
            t = arch_acc(dataset, arch, "true")
            s = arch_acc(dataset, arch, "shuffled")
            if not t or not s:
                continue
            gap = float(t["best_val_acc_mean"]) - float(s["best_val_acc_mean"])
            lines.append(
                f"| `{dataset}` | `{arch}` | {fmt(t['best_val_acc_mean'])} +/- {fmt(t['best_val_acc_std'])} | {fmt(s['best_val_acc_mean'])} +/- {fmt(s['best_val_acc_std'])} | {fmt(gap)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TSception is best treated as a multi-branch fusion-induced reorganization family. Its temporal multi-scale blocks are comparatively moderate and stable, while spatial/fusion/classifier groups begin with high ODI and usually decrease during true-label training. This matches the design goal of combining multi-scale temporal features with spatial/asymmetry fusion.",
            "- ATCNet is a hybrid family. Its convolutional projection starts at low ODI and consistently increases, while its attention and TCN groups start high and usually decrease or reorganize. This gives ATCNet a two-stage HSDD signature: local convolutional collapse plus attention/TCN reorganization.",
            "- Sleep-EDF full is included as the only Sleep-EDF dataset scope. The older Sleep-EDF pilot subset is archived and excluded from this reported-scope taxonomy.",
        ]
    )
    (out_dir / "architecture_taxonomy_all_datasets_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
