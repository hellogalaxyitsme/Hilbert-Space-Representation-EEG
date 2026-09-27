#!/usr/bin/env python3
"""Compare classical filter-bank log-variance baselines with deep-layer HSDD."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
DATASETS = ("sleepedf_full", "seediv", "p300")
SEEDS = (41, 42, 43, 44, 45, 46)
MODES = ("true", "shuffled")
METRICS = ("distance_spearman", "scaled_ratio_error", "odi", "collapse_factor", "local_stretch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/classical_filterbank_dl_comparison")
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--arches", default=" ".join(ARCHES))
    parser.add_argument("--seeds", default=" ".join(str(seed) for seed in SEEDS))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def classical_path(dataset: str, mode: str, seed: int) -> Path:
    return Path(f"reports/{dataset}_filterbank_logvar_hsdd_{mode}_seed{seed}.json")


def dl_path(dataset: str, arch: str, mode: str, seed: int) -> Path:
    return Path(f"reports/{dataset}_{arch}_layer_audit_{mode}_seed{seed}.json")


def metric_value(layer_payload: dict[str, Any], metric: str) -> float:
    if metric == "distance_spearman":
        return float(layer_payload["pairwise_isometry"]["distance_spearman"])
    if metric == "scaled_ratio_error":
        return float(layer_payload["pairwise_isometry"]["scaled_abs_ratio_error_mean"])
    if metric == "odi":
        return float(layer_payload["band_orthogonality"]["output_mean_abs_offdiag_cosine_mean"])
    if metric == "collapse_factor":
        return float(layer_payload["band_orthogonality"]["collapse_factor_mean"])
    if metric == "local_stretch":
        return float(layer_payload["local_directional_distortion"]["stretch_mean"])
    raise ValueError(metric)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


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


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    datasets = tuple(item for item in args.datasets.replace(",", " ").split() if item.strip())
    arches = tuple(item for item in args.arches.replace(",", " ").split() if item.strip())
    seeds = tuple(int(item) for item in args.seeds.replace(",", " ").split() if item.strip())
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for dataset in datasets:
        for mode in MODES:
            for seed in seeds:
                path = classical_path(dataset, mode, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                payload = read_json(path)
                val_acc = float(payload["training"]["val_acc"])
                for layer, layer_payload in payload["layers"].items():
                    for metric in METRICS:
                        rows.append(
                            {
                                "dataset": dataset,
                                "family": "classical_filterbank",
                                "model": "filterbank_logvar_lda",
                                "mode": mode,
                                "seed": seed,
                                "layer": layer,
                                "metric": metric,
                                "value": metric_value(layer_payload, metric),
                                "val_acc": val_acc,
                                "source": str(path),
                            }
                        )
        for arch in arches:
            for mode in MODES:
                for seed in seeds:
                    path = dl_path(dataset, arch, mode, seed)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    payload = read_json(path)
                    val_acc = float(payload["training"]["best_val_acc"])
                    for layer, layer_payload in payload["layers"].items():
                        for metric in METRICS:
                            if metric == "local_stretch" and "local_directional_distortion" not in layer_payload:
                                continue
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "family": "deep_layer",
                                    "model": arch,
                                    "mode": mode,
                                    "seed": seed,
                                    "layer": layer,
                                    "metric": metric,
                                    "value": metric_value(layer_payload, metric),
                                    "val_acc": val_acc,
                                    "source": str(path),
                                }
                            )
    if missing:
        raise FileNotFoundError("missing expected reports:\n" + "\n".join(missing[:100]))

    metric_summary = []
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = {}
    acc_grouped: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["family"], row["model"], row["mode"], row["layer"], row["metric"]), []).append(float(row["value"]))
        acc_grouped.setdefault((row["dataset"], row["family"], row["model"], row["mode"]), {})[int(row["seed"])] = float(row["val_acc"])
    for key, values in sorted(grouped.items()):
        mean, std = mean_std(values)
        metric_summary.append(
            {
                "dataset": key[0],
                "family": key[1],
                "model": key[2],
                "mode": key[3],
                "layer": key[4],
                "metric": key[5],
                "n": len(values),
                "mean": mean,
                "std": std,
                "mean_pm_std": f"{mean:.3f} +/- {std:.3f}",
            }
        )
    acc_rows = []
    for (dataset, family, model, mode), seed_values in sorted(acc_grouped.items()):
        values = [seed_values[seed] for seed in sorted(seed_values)]
        mean, std = mean_std(values)
        acc_rows.append(
            {
                "dataset": dataset,
                "family": family,
                "model": model,
                "mode": mode,
                "n": len(values),
                "val_acc_mean": mean,
                "val_acc_std": std,
                "val_acc_pm_std": f"{mean:.3f} +/- {std:.3f}",
            }
        )

    summary_index = {(r["dataset"], r["family"], r["model"], r["mode"], r["layer"], r["metric"]): r for r in metric_summary}
    vector_rows = []
    layer_keys = sorted({(r["dataset"], r["family"], r["model"], r["mode"], r["layer"]) for r in metric_summary})
    for dataset, family, model, mode, layer in layer_keys:
        vector = {}
        ok = True
        for metric in METRICS:
            row = summary_index.get((dataset, family, model, mode, layer, metric))
            if row is None:
                ok = False
                break
            vector[metric] = float(row["mean"])
        if ok:
            vector_rows.append({"dataset": dataset, "family": family, "model": model, "mode": mode, "layer": layer, **vector})

    nearest_metrics = ("distance_spearman", "scaled_ratio_error", "odi", "local_stretch")
    nearest_rows = []
    for dataset in datasets:
        dataset_vectors = [r for r in vector_rows if r["dataset"] == dataset]
        metric_stds = {}
        for metric in nearest_metrics:
            vals = np.asarray([float(r[metric]) for r in dataset_vectors], dtype=float)
            metric_stds[metric] = float(vals.std(ddof=0)) or 1.0
        targets = [r for r in dataset_vectors if r["family"] == "classical_filterbank" and r["layer"] in {"filterbank_concat", "lda_logits"}]
        candidates = [r for r in dataset_vectors if r["family"] == "deep_layer"]
        for target in targets:
            scored = []
            for cand in candidates:
                if cand["mode"] != target["mode"]:
                    continue
                sq = 0.0
                for metric in nearest_metrics:
                    dz = (float(cand[metric]) - float(target[metric])) / metric_stds[metric]
                    sq += dz * dz
                scored.append((float(np.sqrt(sq)), cand))
            for rank, (distance, cand) in enumerate(sorted(scored, key=lambda item: item[0])[:10], start=1):
                nearest_rows.append(
                    {
                        "dataset": dataset,
                        "target_layer": target["layer"],
                        "mode": target["mode"],
                        "rank": rank,
                        "model": cand["model"],
                        "layer": cand["layer"],
                        "z_distance": distance,
                        **{f"candidate_{metric}": cand[metric] for metric in nearest_metrics},
                        **{f"target_{metric}": target[metric] for metric in nearest_metrics},
                    }
                )

    write_csv(out_dir / "classical_filterbank_dl_layer_metrics.csv", rows)
    write_csv(out_dir / "classical_filterbank_dl_metric_summary.csv", metric_summary)
    write_csv(out_dir / "classical_filterbank_dl_accuracy_summary.csv", acc_rows)
    write_csv(out_dir / "classical_filterbank_nearest_deep_layers.csv", nearest_rows)

    acc_index = {(r["dataset"], r["family"], r["model"], r["mode"]): r for r in acc_rows}
    lines = [
        "# Classical Filter-Bank vs Deep Layer HSDD Comparison",
        "",
        f"Datasets: {', '.join(datasets)}.",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "",
        "Classical model: Butterworth filter bank, per-channel log-variance features, standardized shrinkage LDA.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'classical_filterbank_dl_layer_metrics.csv'}`",
        f"- `{out_dir / 'classical_filterbank_dl_metric_summary.csv'}`",
        f"- `{out_dir / 'classical_filterbank_dl_accuracy_summary.csv'}`",
        f"- `{out_dir / 'classical_filterbank_nearest_deep_layers.csv'}`",
        "",
        "## Accuracy Anchor",
        "",
        "| Dataset | Model | True val acc | Shuffled val acc | Gap |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        true_row = acc_index[dataset, "classical_filterbank", "filterbank_logvar_lda", "true"]
        shuf_row = acc_index[dataset, "classical_filterbank", "filterbank_logvar_lda", "shuffled"]
        gap = float(true_row["val_acc_mean"]) - float(shuf_row["val_acc_mean"])
        lines.append(f"| `{dataset}` | filterbank logvar + LDA | {true_row['val_acc_pm_std']} | {shuf_row['val_acc_pm_std']} | {fmt(gap)} |")
    lines += [
        "",
        "## Classical Geometry",
        "",
        "| Dataset | Layer | Mode | Distance Spearman | Scaled ratio error | ODI | Local stretch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for layer in ("filterbank_concat", "lda_logits"):
            for mode in MODES:
                vals = {}
                for metric in ("distance_spearman", "scaled_ratio_error", "odi", "local_stretch"):
                    row = summary_index[(dataset, "classical_filterbank", "filterbank_logvar_lda", mode, layer, metric)]
                    vals[metric] = row["mean_pm_std"]
                lines.append(
                    f"| `{dataset}` | `{layer}` | {mode} | {vals['distance_spearman']} | "
                    f"{vals['scaled_ratio_error']} | {vals['odi']} | {vals['local_stretch']} |"
                )
    lines += [
        "",
        "## Nearest Deep Layers",
        "",
        "| Dataset | Classical target | Mode | Nearest DL layer | Z-distance |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in nearest_rows:
        if int(row["rank"]) == 1 and row["target_layer"] in {"filterbank_concat", "lda_logits"}:
            lines.append(
                f"| `{row['dataset']}` | `{row['target_layer']}` | {row['mode']} | "
                f"{row['model']} `{row['layer']}` | {fmt(float(row['z_distance']))} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a cross-task classical EEG anchor, not a replacement for BCI IV 2a FBCSP. CSP remains the stronger motor-imagery-specific baseline.",
        "- The filter-bank baseline asks whether DL layers resemble classical band-power geometry on sleep staging, emotion recognition, and P300 detection.",
        "- Collapse-factor values are included in CSVs for completeness but are retained as scale-ratio context alongside ODI and distance-preservation metrics.",
    ]
    (out_dir / "classical_filterbank_dl_comparison_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
