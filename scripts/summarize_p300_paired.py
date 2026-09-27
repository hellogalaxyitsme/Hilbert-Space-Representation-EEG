#!/usr/bin/env python3
"""Regenerate fixed-hook P300 aggregate summaries from compact seed means."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="fixed_hook_layer_seed.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"arch", "layer", "seed", "epoch", "n_windows", "raw_odi", "anchored_odi", "signed_baseline", "signed_cross", "signed_residual"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("input is empty or lacks required fixed-hook columns")
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["arch"], row["layer"], int(row["epoch"]))].append(row)
    output = []
    for (arch, layer, epoch), group in sorted(groups.items()):
        seeds = {row["seed"] for row in group}
        if len(seeds) != len(group):
            raise ValueError(f"duplicate fixed-hook seed record for {(arch, layer, epoch)}")
        output.append({
            "arch": arch, "layer": layer, "epoch": epoch, "n_seed_checkpoints": len(group),
            "n_windows_per_checkpoint": int(group[0]["n_windows"]),
            **{name: sum(float(row[name]) for row in group) / len(group) for name in ("raw_odi", "anchored_odi", "signed_baseline", "signed_cross", "signed_residual")},
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} fixed-hook summaries to {args.out}")


if __name__ == "__main__":
    main()
