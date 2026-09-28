#!/usr/bin/env python3
"""Compare BCI IV 2a FBCSP geometry against deep-layer HSDD reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
SEEDS = (41, 42, 43, 44, 45, 46)
MODES = ("true", "shuffled")

BCI_TRUE_BASE = {
    "eegnet": "reports/eegnet_bci2a_layer_audit.json",
    "shallowconvnet": "reports/shallowconvnet_bci2a_layer_audit.json",
    "eegconformer": "reports/eegconformer_bci2a_layer_audit.json",
    "tsception": "reports/tsception_bci2a_layer_audit_true.json",
    "atcnet": "reports/atcnet_bci2a_layer_audit_true.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/bci2a_fbcsp_dl_comparison")
    parser.add_argument("--seeds", default=" ".join(str(seed) for seed in SEEDS))
    parser.add_argument("--arches", default=" ".join(ARCHES))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dl_static_path(arch: str, mode: str, seed: int) -> Path:
    if seed == 41:
        if mode == "true":
            return Path(BCI_TRUE_BASE[arch])
        return Path(f"reports/{arch}_bci2a_layer_audit_{mode}.json")
    return Path(f"reports/{arch}_bci2a_layer_audit_{mode}_seed{seed}.json")


def fbcsp_path(mode: str, seed: int) -> Path:
    return Path(f"reports/bci2a_fbcsp_hsdd_{mode}_seed{seed}.json")


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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    seeds = tuple(int(item) for item in args.seeds.replace(",", " ").split() if item.strip())
    arches = tuple(item for item in args.arches.replace(",", " ").split() if item.strip())
    metrics = ("distance_spearman", "scaled_ratio_error", "odi", "collapse_factor", "local_stretch")
    rows = []
    missing = []

    for mode in MODES:
        for seed in seeds:
            path = fbcsp_path(mode, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            payload = read_json(path)
            val_acc = float(payload["training"]["val_acc"])
            for layer, layer_payload in payload["layers"].items():
                for metric in metrics:
                    rows.append(
                        {
                            "family": "fbcsp",
                            "model": "fbcsp",
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
                path = dl_static_path(arch, mode, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                payload = read_json(path)
                val_acc = float(payload["training"]["best_val_acc"])
                for layer, layer_payload in payload["layers"].items():
                    for metric in metrics:
                        if metric == "local_stretch" and "local_directional_distortion" not in layer_payload:
                            continue
                        rows.append(
                            {
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

    summary_rows = []
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    acc_grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["model"], row["mode"], row["layer"], row["metric"]), []).append(float(row["value"]))
        acc_grouped.setdefault((row["family"], row["model"], row["mode"]), []).append(float(row["val_acc"]))
    for (family, model, mode, layer, metric), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        summary_rows.append(
            {
                "family": family,
                "model": model,
                "mode": mode,
                "layer": layer,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": std,
                "mean_pm_std": f"{mean:.3f} +/- {std:.3f}",
            }
        )

    acc_rows = []
    for (family, model, mode), values in sorted(acc_grouped.items()):
        # One val_acc repeats per layer/metric in rows; recover unique seed-level values.
        unique = []
        seen = set()
        for row in rows:
            key = (row["family"], row["model"], row["mode"], row["seed"])
            if row["family"] == family and row["model"] == model and row["mode"] == mode and key not in seen:
                unique.append(float(row["val_acc"]))
                seen.add(key)
        mean, std = mean_std(unique)
        acc_rows.append({"family": family, "model": model, "mode": mode, "n": len(unique), "val_acc_mean": mean, "val_acc_std": std})

    write_csv(out_dir / "bci2a_fbcsp_dl_layer_metrics.csv", rows)
    write_csv(out_dir / "bci2a_fbcsp_dl_metric_summary.csv", summary_rows)
    write_csv(out_dir / "bci2a_fbcsp_dl_accuracy_summary.csv", acc_rows)

    summary_index = {
        (row["family"], row["model"], row["mode"], row["layer"], row["metric"]): row
        for row in summary_rows
    }
    vector_rows = []
    layer_keys = sorted({(row["family"], row["model"], row["mode"], row["layer"]) for row in summary_rows})
    for family, model, mode, layer in layer_keys:
        vector = {}
        ok = True
        for metric in metrics:
            row = summary_index.get((family, model, mode, layer, metric))
            if row is None:
                ok = False
                break
            vector[metric] = float(row["mean"])
        if ok:
            vector_rows.append({"family": family, "model": model, "mode": mode, "layer": layer, **vector})

    nearest_metrics = ("distance_spearman", "scaled_ratio_error", "odi", "local_stretch")
    metric_stds = {}
    for metric in nearest_metrics:
        vals = np.asarray([float(row[metric]) for row in vector_rows], dtype=float)
        metric_stds[metric] = float(vals.std(ddof=0)) or 1.0

    nearest_rows = []
    targets = [
        row
        for row in vector_rows
        if row["family"] == "fbcsp" and row["model"] == "fbcsp" and row["layer"] in {"fbcsp_concat", "lda_logits"}
    ]
    candidates = [row for row in vector_rows if row["family"] == "deep_layer"]
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
    write_csv(out_dir / "bci2a_fbcsp_nearest_deep_layers.csv", nearest_rows)

    acc_index = {(row["family"], row["model"], row["mode"]): row for row in acc_rows}
    fbcsp_layers = ["fbcsp_concat", "lda_logits"]
    lines = [
        "# BCI IV 2a FBCSP vs Deep Layer HSDD Comparison",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "",
        "Artifacts:",
        "",
        f"- `{out_dir / 'bci2a_fbcsp_dl_layer_metrics.csv'}`",
        f"- `{out_dir / 'bci2a_fbcsp_dl_metric_summary.csv'}`",
        f"- `{out_dir / 'bci2a_fbcsp_dl_accuracy_summary.csv'}`",
        f"- `{out_dir / 'bci2a_fbcsp_nearest_deep_layers.csv'}`",
        "",
        "## Validation Accuracy",
        "",
        "| Family | Model | True val acc | Shuffled val acc |",
        "| --- | --- | ---: | ---: |",
    ]
    for family, model in [("fbcsp", "fbcsp")] + [("deep_layer", arch) for arch in arches]:
        true = acc_index.get((family, model, "true"))
        shuffled = acc_index.get((family, model, "shuffled"))
        if true and shuffled:
            lines.append(
                f"| {family} | {model} | {true['val_acc_mean']:.3f} +/- {true['val_acc_std']:.3f} | "
                f"{shuffled['val_acc_mean']:.3f} +/- {shuffled['val_acc_std']:.3f} |"
            )
    lines += [
        "",
        "## FBCSP Geometry",
        "",
        "| Layer | Mode | Distance Spearman | Scaled ratio error | ODI | Local stretch |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for layer in fbcsp_layers:
        for mode in MODES:
            vals = {
                metric: summary_index.get(("fbcsp", "fbcsp", mode, layer, metric), {})
                for metric in metrics
            }
            lines.append(
                f"| `{layer}` | {mode} | "
                f"{vals['distance_spearman'].get('mean_pm_std', 'NA')} | "
                f"{vals['scaled_ratio_error'].get('mean_pm_std', 'NA')} | "
                f"{vals['odi'].get('mean_pm_std', 'NA')} | "
                f"{vals['local_stretch'].get('mean_pm_std', 'NA')} |"
            )
    lines += [
        "",
        "## Closest Deep Layers to FBCSP Geometry",
        "",
        "Z-distance is computed over mean distance Spearman, scaled ratio error, ODI, and local stretch. Collapse factor is kept in the full metric table but omitted here because it is unstable when input band components are nearly orthogonal.",
        "",
        "| Target | Mode | Rank | Deep model | Deep layer | Z-distance |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in nearest_rows:
        if row["rank"] <= 5:
            lines.append(
                f"| `{row['target_layer']}` | {row['mode']} | {row['rank']} | "
                f"{row['model']} | `{row['layer']}` | {fmt(row['z_distance'])} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- FBCSP provides an EEG-literature baseline representation and is evaluated as a feature-space baseline.",
        "- `fbcsp_concat` is the concatenated log-variance CSP feature space across the filter bank.",
        "- `lda_logits` is included as the supervised classifier space downstream of CSP.",
        "- The same HSDD metrics used for DL layers are applied here, so CSP can anchor whether DL layers are preserving, discarding, or reorganizing classical EEG band-power geometry.",
        "- The nearest-layer table gives a concrete bridge from classical FBCSP geometry to learned DL geometry; layers that appear there are the strongest candidates for CSP-like representational behavior.",
    ]
    (out_dir / "bci2a_fbcsp_dl_comparison_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
