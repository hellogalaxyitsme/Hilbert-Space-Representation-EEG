#!/usr/bin/env python3
"""Validate and summarize compact fixed-hook SEED-IV audit records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REQUIRED = {"arch", "seed", "epoch", "layer", "raw_odi", "anchored_odi"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError("input lacks compact fixed-hook columns")
    keys = set()
    by_arch_epoch = defaultdict(list)
    for row in rows:
        key = (row["arch"], row["seed"], row["epoch"], row["layer"])
        if key in keys:
            raise ValueError(f"duplicate fixed-hook record: {key}")
        keys.add(key)
        for name in ("raw_odi", "anchored_odi"):
            value = float(row[name])
            if not 0 <= value <= 1:
                raise ValueError(f"{name} outside [0, 1] for {key}")
        by_arch_epoch[(row["arch"], int(row["epoch"]))].append(row)
    output = []
    for (arch, epoch), group in sorted(by_arch_epoch.items()):
        output.append({
            "arch": arch,
            "epoch": epoch,
            "n_fixed_hook_seed_records": len(group),
            "n_seeds": len({row["seed"] for row in group}),
            "n_hooks": len({row["layer"] for row in group}),
            "raw_odi_mean": sum(float(row["raw_odi"]) for row in group) / len(group),
            "anchored_odi_mean": sum(float(row["anchored_odi"]) for row in group) / len(group),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"records": len(rows), "architecture_epochs": len(output), "out": str(args.out)}))


if __name__ == "__main__":
    main()
