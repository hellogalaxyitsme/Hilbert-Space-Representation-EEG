#!/usr/bin/env python3
"""Validate the complete compact fixed-hook SEED-IV audit record."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

ARCH_HOOKS = {
    "eegnet": {"conv_temporal", "bnorm_1", "conv_spatial", "conv_separable_depth", "bnorm_2", "pool_2", "logits"},
    "shallowconvnet": {"conv_time_spat", "bnorm", "conv_nonlin_exp", "pool", "pool_nonlin_exp", "logits"},
    "eegconformer": {"patch_embedding", "transformer.0", "transformer.2", "transformer.5", "fc", "logits"},
    "tsception": {"temporal_blocks.0", "temporal_blocks.2", "spatial_block_1", "spatial_block_2", "batch_temporal_lay", "batch_spatial_lay", "dense_layer", "logits"},
    "atcnet": {"temporal_conv_nets.0", "temporal_conv_nets.4", "attention_blocks.0", "attention_blocks.4", "conv_block", "final_layer.4", "logits"},
}
SEEDS = tuple(range(41, 47))
EPOCHS = (0, 1, 2, 5, 10, 20)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def parse_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {row!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--fixed-hooks", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary JSON must be an object")
    coverage = read_csv(args.coverage)
    rows = read_csv(args.fixed_hooks)
    expected_checkpoints = {(arch, seed, epoch) for arch in ARCH_HOOKS for seed in SEEDS for epoch in EPOCHS}
    coverage_keys = {(row.get("arch"), parse_int(row, "seed"), parse_int(row, "epoch")) for row in coverage}
    if len(coverage) != len(expected_checkpoints) or coverage_keys != expected_checkpoints:
        raise ValueError("coverage must contain each architecture, seed, and epoch exactly once")
    for row in coverage:
        expected_layers = len(ARCH_HOOKS[row["arch"]])
        if parse_int(row, "n_layers") != expected_layers or parse_int(row, "n_windows") != 64 or parse_int(row, "n_rows") != 64 * expected_layers:
            raise ValueError(f"coverage counts do not match fixed-hook scope: {row['arch']}")
    expected_hooks = {(arch, seed, epoch, layer) for arch, seed, epoch in expected_checkpoints for layer in ARCH_HOOKS[arch]}
    hook_keys = {(row.get("arch"), parse_int(row, "seed"), parse_int(row, "epoch"), row.get("layer")) for row in rows}
    if len(rows) != len(expected_hooks) or hook_keys != expected_hooks:
        raise ValueError("fixed-hook rows must have exact architecture, seed, epoch, and layer identities")
    for row in rows:
        for field in ("raw_odi", "anchored_odi"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {field}: {row!r}") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {field}: {row!r}")
    if summary.get("observed_checkpoints") != len(expected_checkpoints) or summary.get("fixed_hook_rows") != len(expected_hooks):
        raise ValueError("summary counts do not match the compact records")
    print(json.dumps({"checkpoints": len(coverage_keys), "fixed_hook_records": len(hook_keys), "per_window_rows_not_distributed": summary.get("per_window_rows")}))


if __name__ == "__main__":
    main()
