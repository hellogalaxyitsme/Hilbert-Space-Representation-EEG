#!/usr/bin/env python3
"""Diagnose why Tier-3 taxonomy-conditioned HSDD did or did not beat ODI minimization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from hsdd.io import write_json
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import load_bci2a, make_train_val, set_seed
from run_tier3_bci2a_regularizer import (
    TAXONOMY_TARGETS,
    GradLayerCapture,
    smooth_odi_from_features,
    torch_band_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/tier3_regularizer_bci2a")
    parser.add_argument("--stats-dir", default="reports/tier3_regularizer_bci2a_stats")
    parser.add_argument("--out-dir", default="reports/tier3_regularizer_bci2a_diagnostics")
    parser.add_argument("--root", default=None)
    parser.add_argument("--n-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--skip-checkpoint-probe", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
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


def finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def fmt(x: Any, digits: int = 4) -> str:
    val = finite_float(x)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def bootstrap_ci(vals: list[float], *, n_boot: int = 5000, seed: int = 99) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def report_rows(reports_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_dir.glob("*.json")):
        payload = read_json(path)
        history = payload.get("history", [])
        reg_losses = [finite_float(r.get("train_regularizer_loss")) for r in history[-5:]]
        task_losses = [finite_float(r.get("train_task_loss")) for r in history[-5:]]
        reg_mean, _ = mean_std(reg_losses)
        task_mean, _ = mean_std(task_losses)
        endpoints = payload["endpoints"]
        run = payload["run"]
        rows.append(
            {
                "dataset": payload["dataset"]["name"],
                "arch": payload["model"]["arch"],
                "arm": run["arm"],
                "seed": int(run["seed"]),
                "train_fraction": float(payload["dataset"]["train_fraction"]),
                "lambda_hsdd": finite_float(run.get("lambda_hsdd")),
                "reg_every": int(run.get("reg_every", -1)),
                "reg_batch_size": int(run.get("reg_batch_size", -1)),
                "target_scheme": str(run.get("target_scheme", payload.get("model", {}).get("target_metadata", {}).get("scheme", "hand_taxon"))),
                "target_file": str(run.get("target_file", payload.get("model", {}).get("target_metadata", {}).get("source", "")) or ""),
                "clean_acc": endpoints["clean"]["acc"],
                "clean_ece": endpoints["clean"]["ece"],
                "band_noise_mean_acc": endpoints["band_limited_noise"]["mean_acc"],
                "band_noise_mean_acc_drop": endpoints["band_limited_noise"]["mean_acc_drop"],
                "channel_dropout_mean_acc": endpoints["channel_dropout"]["mean_acc"],
                "channel_dropout_mean_acc_drop": endpoints["channel_dropout"]["mean_acc_drop"],
                "last5_regularizer_loss_mean": reg_mean,
                "last5_task_loss_mean": task_mean,
                "regularizer_to_task_ratio": reg_mean / task_mean if task_mean and math.isfinite(task_mean) else float("nan"),
                "checkpoint": run["checkpoint"],
                "source": str(path),
            }
        )
    return rows


@torch.no_grad()
def checkpoint_layer_odi(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    arch: str,
    layer_names: tuple[str, ...],
    sfreq: float,
    device: torch.device,
) -> dict[str, float]:
    components, _ = torch_band_components(torch.from_numpy(x.astype(np.float32)).to(device), sfreq=sfreq)
    batch, n_bands = components.shape[:2]
    band_batch = components.reshape(batch * n_bands, *components.shape[2:])
    capture = GradLayerCapture(model.eval(), layer_names)
    try:
        outputs = capture.forward(band_batch)
    finally:
        capture.close()
    values = {}
    for name in layer_names:
        if name not in outputs:
            continue
        z = outputs[name].reshape(batch, n_bands, *outputs[name].shape[1:])
        values[name] = float(smooth_odi_from_features(z).detach().cpu())
    return values


def run_checkpoint_probe(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.skip_checkpoint_probe:
        return []
    wanted = [r for r in rows if r["arm"] in {"odi_min", "taxon_hsdd"}]
    if not wanted:
        return []
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=0.0, tmax=3.0)
    _, _, val_x, _ = make_train_val(data, label_mode="true", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val_x), size=min(args.n_epochs, len(val_x)), replace=False)
    probe_x = val_x[idx]
    sfreq = 1.0 / data["dt"]

    out_rows = []
    reports_dir = Path(args.reports_dir)
    for row in wanted:
        arch = row["arch"]
        model, all_layers, _ = arch_spec(arch, probe_x.shape[1], probe_x.shape[2], n_outputs=4, sfreq=sfreq)
        checkpoint = Path(str(row["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.exists():
            checkpoint = reports_dir / "checkpoints" / checkpoint.name
        if not checkpoint.exists():
            print(f"missing checkpoint {checkpoint}", flush=True)
            continue
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        layers = compact_layers(arch, all_layers)
        values = checkpoint_layer_odi(model, probe_x, arch=arch, layer_names=layers, sfreq=sfreq, device=device)
        run_payload = read_json(Path(row["source"]))
        taxon_targets = run_payload.get("model", {}).get("taxonomy_targets") or TAXONOMY_TARGETS.get(arch, {})
        for layer, odi in values.items():
            taxon_target = float(taxon_targets.get(layer, 0.0))
            out_rows.append(
                {
                    "dataset": row["dataset"],
                    "arch": arch,
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "train_fraction": row["train_fraction"],
                    "lambda_hsdd": row["lambda_hsdd"],
                    "reg_every": row["reg_every"],
                    "reg_batch_size": row["reg_batch_size"],
                    "target_scheme": row["target_scheme"],
                    "target_file": row["target_file"],
                    "layer": layer,
                    "posthoc_odi_surrogate": odi,
                    "taxon_target": taxon_target,
                    "agnostic_target": 0.0,
                    "abs_error_to_taxon_target": abs(odi - taxon_target),
                    "abs_error_to_agnostic_target": abs(odi),
                    "checkpoint": str(checkpoint),
                }
            )
        print(
            f"probed {arch} {row['arm']} seed={row['seed']} "
            f"fraction={row['train_fraction']} lambda={row['lambda_hsdd']} "
            f"reg_every={row['reg_every']} reg_batch={row['reg_batch_size']} "
            f"target={row['target_scheme']}",
            flush=True,
        )
    return out_rows


def pair_endpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, float, int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[
            (
                row["arch"],
                int(row["seed"]),
                float(row["train_fraction"]),
                float(row["lambda_hsdd"]),
                int(row["reg_every"]),
                int(row["reg_batch_size"]),
                str(row.get("target_scheme", "hand_taxon")),
            )
        ][row["arm"]] = row
    pairs = []
    for (arch, seed, frac, lambda_hsdd, reg_every, reg_batch_size, target_scheme), vals in sorted(grouped.items()):
        if "odi_min" not in vals or "taxon_hsdd" not in vals:
            continue
        a2 = vals["odi_min"]
        a4 = vals["taxon_hsdd"]
        pairs.append(
            {
                "arch": arch,
                "seed": seed,
                "train_fraction": frac,
                "lambda_hsdd": lambda_hsdd,
                "reg_every": reg_every,
                "reg_batch_size": reg_batch_size,
                "target_scheme": target_scheme,
                "delta_clean_acc_arm4_minus_arm2": finite_float(a4["clean_acc"]) - finite_float(a2["clean_acc"]),
                "delta_clean_ece_arm4_minus_arm2": finite_float(a4["clean_ece"]) - finite_float(a2["clean_ece"]),
                "delta_band_noise_acc_arm4_minus_arm2": finite_float(a4["band_noise_mean_acc"]) - finite_float(a2["band_noise_mean_acc"]),
                "delta_band_noise_drop_arm4_minus_arm2": finite_float(a4["band_noise_mean_acc_drop"]) - finite_float(a2["band_noise_mean_acc_drop"]),
                "delta_channel_dropout_acc_arm4_minus_arm2": finite_float(a4["channel_dropout_mean_acc"]) - finite_float(a2["channel_dropout_mean_acc"]),
                "delta_channel_dropout_drop_arm4_minus_arm2": finite_float(a4["channel_dropout_mean_acc_drop"]) - finite_float(a2["channel_dropout_mean_acc_drop"]),
                "delta_regularizer_to_task_ratio_arm4_minus_arm2": finite_float(a4["regularizer_to_task_ratio"]) - finite_float(a2["regularizer_to_task_ratio"]),
            }
        )
    return pairs


def pair_probe_rows(probe_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, float, float, int, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in probe_rows:
        grouped[
            (
                row["arch"],
                int(row["seed"]),
                float(row["train_fraction"]),
                float(row["lambda_hsdd"]),
                int(row["reg_every"]),
                int(row["reg_batch_size"]),
                str(row.get("target_scheme", "hand_taxon")),
                row["layer"],
            )
        ][row["arm"]] = row
    layer_pairs = []
    for (arch, seed, frac, lambda_hsdd, reg_every, reg_batch_size, target_scheme, layer), vals in sorted(grouped.items()):
        if "odi_min" not in vals or "taxon_hsdd" not in vals:
            continue
        a2 = vals["odi_min"]
        a4 = vals["taxon_hsdd"]
        layer_pairs.append(
            {
                "arch": arch,
                "seed": seed,
                "train_fraction": frac,
                "lambda_hsdd": lambda_hsdd,
                "reg_every": reg_every,
                "reg_batch_size": reg_batch_size,
                "target_scheme": target_scheme,
                "layer": layer,
                "arm2_odi": a2["posthoc_odi_surrogate"],
                "arm4_odi": a4["posthoc_odi_surrogate"],
                "taxon_target": a4["taxon_target"],
                "delta_odi_arm4_minus_arm2": finite_float(a4["posthoc_odi_surrogate"]) - finite_float(a2["posthoc_odi_surrogate"]),
                "delta_abs_error_to_taxon_arm4_minus_arm2": finite_float(a4["abs_error_to_taxon_target"]) - finite_float(a2["abs_error_to_taxon_target"]),
                "delta_abs_error_to_agnostic_arm4_minus_arm2": finite_float(a4["abs_error_to_agnostic_target"]) - finite_float(a2["abs_error_to_agnostic_target"]),
            }
        )

    run_group: dict[tuple[str, int, float, float, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in layer_pairs:
        run_group[
            (
                row["arch"],
                int(row["seed"]),
                float(row["train_fraction"]),
                float(row["lambda_hsdd"]),
                int(row["reg_every"]),
                int(row["reg_batch_size"]),
                str(row.get("target_scheme", "hand_taxon")),
            )
        ].append(row)
    run_pairs = []
    for (arch, seed, frac, lambda_hsdd, reg_every, reg_batch_size, target_scheme), vals in sorted(run_group.items()):
        tax_err = [finite_float(v["delta_abs_error_to_taxon_arm4_minus_arm2"]) for v in vals]
        agn_err = [finite_float(v["delta_abs_error_to_agnostic_arm4_minus_arm2"]) for v in vals]
        odi_delta = [finite_float(v["delta_odi_arm4_minus_arm2"]) for v in vals]
        run_pairs.append(
            {
                "arch": arch,
                "seed": seed,
                "train_fraction": frac,
                "lambda_hsdd": lambda_hsdd,
                "reg_every": reg_every,
                "reg_batch_size": reg_batch_size,
                "target_scheme": target_scheme,
                "n_layers": len(vals),
                "mean_delta_odi_arm4_minus_arm2": float(np.mean(odi_delta)),
                "mean_delta_abs_error_to_taxon_arm4_minus_arm2": float(np.mean(tax_err)),
                "fraction_layers_arm4_closer_to_taxon": float(np.mean(np.asarray(tax_err) < 0.0)),
                "mean_delta_abs_error_to_agnostic_arm4_minus_arm2": float(np.mean(agn_err)),
            }
        )
    return layer_pairs, run_pairs


def summarize_scope(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scopes[("all", "all")].append(row)
        scopes[(row["arch"], "all_fractions")].append(row)
        scopes[(row["arch"], f"fraction_{float(row['train_fraction']):g}")].append(row)
        if "lambda_hsdd" in row:
            config = (
                f"lambda_{float(row['lambda_hsdd']):g}"
                f"_every_{int(row['reg_every'])}"
                f"_regbatch_{int(row['reg_batch_size'])}"
            )
            scopes[("all", config)].append(row)
            scopes[(row["arch"], config)].append(row)
            if "target_scheme" in row:
                target_config = f"target_{row['target_scheme']}_{config}"
                scopes[("all", target_config)].append(row)
                scopes[(row["arch"], target_config)].append(row)
    out = []
    for (arch, scope), vals in sorted(scopes.items()):
        summary = {"arch": arch, "scope": scope, "n": len(vals)}
        for metric in metrics:
            arr = [finite_float(v[metric]) for v in vals]
            mean, std = mean_std(arr)
            lo, hi = bootstrap_ci(arr)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci_low"] = lo
            summary[f"{metric}_ci_high"] = hi
        out.append(summary)
    return out


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    rows = report_rows(reports_dir)
    endpoint_pairs = pair_endpoint_rows(rows)
    endpoint_summary = summarize_scope(
        endpoint_pairs,
        (
            "delta_band_noise_acc_arm4_minus_arm2",
            "delta_band_noise_drop_arm4_minus_arm2",
            "delta_channel_dropout_acc_arm4_minus_arm2",
            "delta_channel_dropout_drop_arm4_minus_arm2",
            "delta_clean_ece_arm4_minus_arm2",
            "delta_clean_acc_arm4_minus_arm2",
            "delta_regularizer_to_task_ratio_arm4_minus_arm2",
        ),
    )
    probe_rows = run_checkpoint_probe(args, rows)
    layer_pairs, run_probe_pairs = pair_probe_rows(probe_rows)
    probe_summary = summarize_scope(
        run_probe_pairs,
        (
            "mean_delta_odi_arm4_minus_arm2",
            "mean_delta_abs_error_to_taxon_arm4_minus_arm2",
            "fraction_layers_arm4_closer_to_taxon",
            "mean_delta_abs_error_to_agnostic_arm4_minus_arm2",
        ),
    ) if run_probe_pairs else []

    joined = []
    probe_by_key = {
        (
            r["arch"],
            int(r["seed"]),
            float(r["train_fraction"]),
            float(r["lambda_hsdd"]),
            int(r["reg_every"]),
            int(r["reg_batch_size"]),
            str(r.get("target_scheme", "hand_taxon")),
        ): r
        for r in run_probe_pairs
    }
    for row in endpoint_pairs:
        key = (
            row["arch"],
            int(row["seed"]),
            float(row["train_fraction"]),
            float(row["lambda_hsdd"]),
            int(row["reg_every"]),
            int(row["reg_batch_size"]),
            str(row.get("target_scheme", "hand_taxon")),
        )
        joined.append(
            {
                **row,
                **{
                    f"probe_{k}": v
                    for k, v in probe_by_key.get(key, {}).items()
                    if k not in {"arch", "seed", "train_fraction", "lambda_hsdd", "reg_every", "reg_batch_size", "target_scheme"}
                },
            }
        )

    write_csv(out_dir / "tier3_diagnostic_run_rows.csv", rows)
    write_csv(out_dir / "tier3_diagnostic_endpoint_pairs.csv", endpoint_pairs)
    write_csv(out_dir / "tier3_diagnostic_endpoint_summary.csv", endpoint_summary)
    write_csv(out_dir / "tier3_posthoc_layer_odi_rows.csv", probe_rows)
    write_csv(out_dir / "tier3_posthoc_layer_pair_deltas.csv", layer_pairs)
    write_csv(out_dir / "tier3_posthoc_run_target_alignment.csv", run_probe_pairs)
    write_csv(out_dir / "tier3_posthoc_target_alignment_summary.csv", probe_summary)
    write_csv(out_dir / "tier3_endpoint_target_joined.csv", joined)

    overall_endpoint = next((r for r in endpoint_summary if r["arch"] == "all" and r["scope"] == "all"), None)
    overall_probe = next((r for r in probe_summary if r["arch"] == "all" and r["scope"] == "all"), None)
    reg_ratios = [finite_float(r["regularizer_to_task_ratio"]) for r in rows if r["arm"] in {"odi_min", "taxon_hsdd"}]
    reg_ratio_mean, reg_ratio_std = mean_std(reg_ratios)
    payload = {
        "coverage": {
            "run_jsons": len(rows),
            "endpoint_pairs": len(endpoint_pairs),
            "posthoc_layer_rows": len(probe_rows),
            "posthoc_layer_pairs": len(layer_pairs),
            "posthoc_run_pairs": len(run_probe_pairs),
        },
        "overall_endpoint": overall_endpoint,
        "overall_target_alignment": overall_probe,
        "regularizer_to_task_ratio_mean": reg_ratio_mean,
        "regularizer_to_task_ratio_std": reg_ratio_std,
    }
    write_json(out_dir / "tier3_diagnostic_summary.json", payload)

    lines = [
        "# Tier-3 BCI IV 2a Regularizer Diagnostic",
        "",
        "Question: why did taxonomy-conditioned HSDD not clearly beat architecture-agnostic ODI minimization?",
        "",
        "## Coverage",
        "",
        f"- Run JSONs analyzed: {len(rows)}",
        f"- Arm-4-vs-arm-2 endpoint pairs: {len(endpoint_pairs)}",
        f"- Post-hoc checkpoint layer rows: {len(probe_rows)}",
        f"- Post-hoc arm-4-vs-arm-2 layer pairs: {len(layer_pairs)}",
        f"- Post-hoc arm-4-vs-arm-2 run pairs: {len(run_probe_pairs)}",
        "",
        "## Endpoint Result",
        "",
    ]
    if overall_endpoint:
        lines += [
            f"- Overall Î” band-noise accuracy, arm 4 minus arm 2: {fmt(overall_endpoint['delta_band_noise_acc_arm4_minus_arm2_mean'])} "
            f"95% CI [{fmt(overall_endpoint['delta_band_noise_acc_arm4_minus_arm2_ci_low'])}, {fmt(overall_endpoint['delta_band_noise_acc_arm4_minus_arm2_ci_high'])}].",
            f"- Overall Î” channel-dropout accuracy, arm 4 minus arm 2: {fmt(overall_endpoint['delta_channel_dropout_acc_arm4_minus_arm2_mean'])} "
            f"95% CI [{fmt(overall_endpoint['delta_channel_dropout_acc_arm4_minus_arm2_ci_low'])}, {fmt(overall_endpoint['delta_channel_dropout_acc_arm4_minus_arm2_ci_high'])}].",
            f"- Overall Î” ECE, arm 4 minus arm 2: {fmt(overall_endpoint['delta_clean_ece_arm4_minus_arm2_mean'])} "
            f"95% CI [{fmt(overall_endpoint['delta_clean_ece_arm4_minus_arm2_ci_low'])}, {fmt(overall_endpoint['delta_clean_ece_arm4_minus_arm2_ci_high'])}].",
        ]
    lines += [
        "",
        "## Target-Adherence Diagnostic",
        "",
    ]
    if overall_probe:
        lines += [
            f"- Mean Î” post-hoc ODI, arm 4 minus arm 2: {fmt(overall_probe['mean_delta_odi_arm4_minus_arm2_mean'])}.",
            f"- Mean Î” absolute error to taxon targets, arm 4 minus arm 2: {fmt(overall_probe['mean_delta_abs_error_to_taxon_arm4_minus_arm2_mean'])} "
            f"95% CI [{fmt(overall_probe['mean_delta_abs_error_to_taxon_arm4_minus_arm2_ci_low'])}, {fmt(overall_probe['mean_delta_abs_error_to_taxon_arm4_minus_arm2_ci_high'])}].",
            f"- Fraction of audited layers where arm 4 is closer to the taxon target than arm 2: {fmt(overall_probe['fraction_layers_arm4_closer_to_taxon_mean'])}.",
        ]
    else:
        lines.append("- Post-hoc checkpoint probing was skipped or produced no paired rows.")
    lines += [
        "",
        "## Regularizer Scale",
        "",
        f"- Mean last-five-epoch regularizer/task-loss ratio for ODI/HSDD arms: {fmt(reg_ratio_mean, 6)} +/- {fmt(reg_ratio_std, 6)}.",
        "",
        "## Diagnosis",
        "",
    ]
    if overall_endpoint and overall_probe:
        endpoint_bad = finite_float(overall_endpoint["delta_band_noise_acc_arm4_minus_arm2_mean"]) <= 0.0 and finite_float(overall_endpoint["delta_channel_dropout_acc_arm4_minus_arm2_mean"]) <= 0.0
        target_closer = finite_float(overall_probe["mean_delta_abs_error_to_taxon_arm4_minus_arm2_mean"]) < 0.0
        target_reliably_closer = target_closer and finite_float(overall_probe["mean_delta_abs_error_to_taxon_arm4_minus_arm2_ci_high"]) < 0.0
        ratio_tiny = finite_float(reg_ratio_mean) < 0.005
        if endpoint_bad and target_reliably_closer:
            lines.append("- Arm 4 reliably moves representations toward the requested taxon targets without a corresponding robustness improvement.")
        elif endpoint_bad and target_closer:
            lines.append("- Arm 4 is only weakly closer to the requested taxon targets: the mean target-error delta favors arm 4, the confidence interval crosses zero, and about half of audited layers improve.")
        elif endpoint_bad and not target_closer:
            lines.append("- Arm 4 does not reliably reach the taxon targets and does not improve robustness in this configuration.")
        if ratio_tiny:
            lines.append("- The regularizer-to-task-loss ratio is small in this configuration.")
        else:
            lines.append("- The regularizer has non-negligible scale, so the negative endpoint result should be taken seriously for this configuration.")
    lines += [
        "",
        "The reported diagnostics are limited to the configured BCI IV 2a training settings.",
    ]
    (out_dir / "tier3_diagnostic_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
