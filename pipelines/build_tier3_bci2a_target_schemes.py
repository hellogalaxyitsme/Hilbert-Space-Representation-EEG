#!/usr/bin/env python3
"""Build recalibrated Tier-3 BCI IV 2a HSDD target schemes.

The target files produced here are consumed by
`run_tier3_bci2a_regularizer.py --target-file`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPACT_LAYERS: dict[str, tuple[str, ...]] = {
    "eegnet": (
        "conv_temporal",
        "conv_spatial",
        "bnorm_1",
        "conv_separable_depth",
        "bnorm_2",
        "pool_2",
        "logits",
    ),
    "shallowconvnet": (
        "conv_time_spat",
        "bnorm",
        "conv_nonlin_exp",
        "pool",
        "pool_nonlin_exp",
        "logits",
    ),
    "eegconformer": (
        "patch_embedding",
        "transformer.0",
        "transformer.2",
        "transformer.5",
        "fc",
        "logits",
    ),
    "tsception": (
        "temporal_blocks.0",
        "temporal_blocks.2",
        "batch_temporal_lay",
        "spatial_block_1",
        "spatial_block_2",
        "batch_spatial_lay",
        "dense_layer",
        "logits",
    ),
    "atcnet": (
        "conv_block",
        "attention_blocks.0",
        "attention_blocks.4",
        "temporal_conv_nets.0",
        "temporal_conv_nets.4",
        "final_layer.4",
        "logits",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="reports/tier3_bci2a_target_schemes")
    parser.add_argument(
        "--taxonomy-rows",
        default="reports/architecture_taxonomy_all_datasets/architecture_taxonomy_layer_rows.csv",
    )
    parser.add_argument(
        "--matched-null-rows",
        default="reports/tier2_stats_ckpt_extended/matched_null_layer_rows.csv",
    )
    parser.add_argument(
        "--diagnostic-run-rows",
        default="reports/tier3_regularizer_bci2a_lambda_schedule_stats/tier3_diagnostic_run_rows.csv",
    )
    parser.add_argument(
        "--diagnostic-layer-rows",
        default="reports/tier3_regularizer_bci2a_lambda_schedule_stats/tier3_posthoc_layer_odi_rows.csv",
    )
    return parser.parse_args()


def finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def mean(vals: list[float]) -> float:
    good = [v for v in vals if math.isfinite(v)]
    if not good:
        return float("nan")
    return float(sum(good) / len(good))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_hand_targets(project_root: Path) -> dict[str, dict[str, float]]:
    source = PROJECT_ROOT / "pipelines" / "run_tier3_bci2a_regularizer.py"
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TAXONOMY_TARGETS":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TAXONOMY_TARGETS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"could not find TAXONOMY_TARGETS in {source}")


def clamp_odi(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def complete_targets(
    raw: dict[str, dict[str, float]],
    hand_targets: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    targets: dict[str, dict[str, float]] = {}
    fallbacks: dict[str, list[str]] = {}
    for arch, layers in COMPACT_LAYERS.items():
        arch_targets = {}
        missing = []
        for layer in layers:
            val = clamp_odi(raw.get(arch, {}).get(layer, float("nan")))
            if not math.isfinite(val):
                val = float(hand_targets[arch][layer])
                missing.append(layer)
            arch_targets[layer] = val
        targets[arch] = arch_targets
        if missing:
            fallbacks[arch] = missing
    return targets, fallbacks


def build_init_epoch0(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != "bci2a" or row.get("mode") != "true":
            continue
        grouped[(row["arch"], row["layer"])].append(finite_float(row.get("odi_epoch0")))
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (arch, layer), vals in grouped.items():
        out[arch][layer] = mean(vals)
    return {arch: dict(vals) for arch, vals in out.items()}


def build_matched_null(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != "bci2a":
            continue
        if row.get("mode") != "true" or row.get("band_scheme") != "canonical":
            continue
        if str(row.get("epoch")) != "30":
            continue
        grouped[(row["arch"], row["layer"])].append(finite_float(row.get("matched_null_odi_mean")))
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (arch, layer), vals in grouped.items():
        out[arch][layer] = mean(vals)
    return {arch: dict(vals) for arch, vals in out.items()}


def robustness_score(row: dict[str, str]) -> float:
    band = finite_float(row.get("band_noise_mean_acc"))
    drop = finite_float(row.get("channel_dropout_mean_acc"))
    ece = finite_float(row.get("clean_ece"), 0.0)
    return band + drop - max(0.0, ece)


def run_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["arch"],
        str(row["seed"]),
        str(float(row["train_fraction"])),
        str(float(row["lambda_hsdd"])),
        str(int(float(row["reg_every"]))),
        str(int(float(row["reg_batch_size"]))),
    )


def build_robust_checkpoint(
    run_rows: list[dict[str, str]],
    layer_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    best: dict[str, dict[str, str]] = {}
    for row in run_rows:
        if row.get("dataset") != "bci2a" or row.get("arm") != "taxon_hsdd":
            continue
        scheme = row.get("target_scheme", "hand_taxon")
        if scheme != "hand_taxon":
            continue
        arch = row["arch"]
        if arch not in best or robustness_score(row) > robustness_score(best[arch]):
            best[arch] = row

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in layer_rows:
        if row.get("dataset") != "bci2a" or row.get("arm") != "taxon_hsdd":
            continue
        arch = row["arch"]
        if arch not in best:
            continue
        if run_key(row) != run_key(best[arch]):
            continue
        values[(arch, row["layer"])].append(finite_float(row.get("posthoc_odi_surrogate")))

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (arch, layer), vals in values.items():
        out[arch][layer] = mean(vals)

    selected = {
        arch: {
            "seed": int(best_row["seed"]),
            "lambda_hsdd": finite_float(best_row["lambda_hsdd"]),
            "reg_every": int(float(best_row["reg_every"])),
            "reg_batch_size": int(float(best_row["reg_batch_size"])),
            "band_noise_mean_acc": finite_float(best_row["band_noise_mean_acc"]),
            "channel_dropout_mean_acc": finite_float(best_row["channel_dropout_mean_acc"]),
            "clean_ece": finite_float(best_row["clean_ece"]),
            "score": robustness_score(best_row),
            "source": best_row.get("source", ""),
        }
        for arch, best_row in best.items()
    }
    return {arch: dict(vals) for arch, vals in out.items()}, selected


def write_payload(
    out_dir: Path,
    *,
    scheme: str,
    description: str,
    raw_targets: dict[str, dict[str, float]],
    hand_targets: dict[str, dict[str, float]],
    sources: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets, fallbacks = complete_targets(raw_targets, hand_targets)
    payload = {
        "metadata": {
            "scheme": scheme,
            "dataset": "bci2a",
            "description": description,
            "sources": sources,
            "fallback_to_hand_taxon_layers": fallbacks,
            **(extra_metadata or {}),
        },
        "targets": targets,
    }
    path = out_dir / f"{scheme}_targets.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_summary(out_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    rows = []
    for scheme, payload in payloads.items():
        for arch, layer_vals in payload["targets"].items():
            for layer, value in layer_vals.items():
                rows.append({"scheme": scheme, "arch": arch, "layer": layer, "target_odi": value})
    csv_path = out_dir / "target_scheme_values.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheme", "arch", "layer", "target_odi"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Tier-3 BCI IV 2a Target Scheme Recalibration",
        "",
        "These target maps replace the hand-operationalized taxonomy targets in the taxonomy-conditioned HSDD arm.",
        "",
        "| Scheme | Definition | Fallback layers |",
        "| --- | --- | ---: |",
    ]
    for scheme, payload in payloads.items():
        fallbacks = payload["metadata"].get("fallback_to_hand_taxon_layers", {})
        n_fallback = sum(len(v) for v in fallbacks.values())
        lines.append(
            f"| `{scheme}` | {payload['metadata']['description']} | {n_fallback} |"
        )
    lines += [
        "",
        "The `robust_checkpoint` scheme derives from previous robustness endpoints and evaluates target validity under that scope. A primary causal test requires a new training-only selection protocol.",
        "",
        f"Layer target values: `{csv_path}`.",
    ]
    (out_dir / "target_scheme_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_path = project_root / args.taxonomy_rows
    matched_path = project_root / args.matched_null_rows
    run_path = project_root / args.diagnostic_run_rows
    layer_path = project_root / args.diagnostic_layer_rows
    hand_targets = load_hand_targets(project_root)

    robust_raw, robust_selected = build_robust_checkpoint(read_csv(run_path), read_csv(layer_path))
    payloads = {
        "hand_taxon": write_payload(
            out_dir,
            scheme="hand_taxon",
            description="Original hand-operationalized targets from the architecture taxonomy.",
            raw_targets=hand_targets,
            hand_targets=hand_targets,
            sources={"runner": "scripts/run_tier3_bci2a_regularizer.py"},
        ),
        "init_epoch0": write_payload(
            out_dir,
            scheme="init_epoch0",
            description="Mean epoch-0 true-label ODI for each compact layer across available BCI IV 2a seeds.",
            raw_targets=build_init_epoch0(read_csv(taxonomy_path)),
            hand_targets=hand_targets,
            sources={"taxonomy_rows": str(args.taxonomy_rows)},
        ),
        "matched_null": write_payload(
            out_dir,
            scheme="matched_null",
            description="Mean final-checkpoint matched-null ODI for true-label canonical-band BCI IV 2a rows; this targets zero excess over the dimension-matched null in raw ODI units.",
            raw_targets=build_matched_null(read_csv(matched_path)),
            hand_targets=hand_targets,
            sources={"matched_null_rows": str(args.matched_null_rows)},
        ),
        "robust_checkpoint": write_payload(
            out_dir,
            scheme="robust_checkpoint",
            description="Layer ODIs from the previously strongest hand-taxon HSDD checkpoint per architecture by band-noise plus channel-dropout accuracy minus ECE.",
            raw_targets=robust_raw,
            hand_targets=hand_targets,
            sources={
                "diagnostic_run_rows": str(args.diagnostic_run_rows),
                "diagnostic_layer_rows": str(args.diagnostic_layer_rows),
            },
            extra_metadata={"selected_checkpoints": robust_selected, "exploratory": True},
        ),
    }
    write_summary(out_dir, payloads)
    print(f"wrote target schemes to {out_dir}")


if __name__ == "__main__":
    main()
