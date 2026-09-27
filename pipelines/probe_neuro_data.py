#!/usr/bin/env python3
"""Inventory candidate small EEG datasets without reading raw signal payloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsdd.io import write_json


PATTERNS = ("*.gdf", "*.edf", "*.fif", "*.mat", "*.set", "*.vhdr", "*.npz", "*.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="reports/neuro_data_probe.json")
    parser.add_argument("--max-files", type=int, default=12)
    return parser.parse_args()


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    candidates = [
        root / "Jasmeet_Datasets" / "BCI IV 2a",
        root / "Jasmeet_Datasets" / "BCI_IV_2a_GDF_to_FIF",
        root / "BCI IV 2a Dataset",
        root / "Jasmeet_Datasets" / "BNCI2014_009",
        root / "Jasmeet_Datasets" / "SEED IV",
        root / "DREAMER",
        root / "DEAP",
    ]
    report = {"root": str(root), "candidates": []}
    for candidate in candidates:
        entry = {
            "path": str(candidate),
            "exists": candidate.exists(),
            "file_counts": {},
            "examples": [],
        }
        if candidate.exists():
            for pattern in PATTERNS:
                entry["file_counts"][pattern] = count_files(candidate, pattern)
            examples = []
            for pattern in PATTERNS:
                for file_path in sorted(candidate.rglob(pattern))[: args.max_files]:
                    examples.append(str(file_path))
                if len(examples) >= args.max_files:
                    break
            entry["examples"] = examples[: args.max_files]
        report["candidates"].append(entry)

    write_json(args.out, report)
    print(f"wrote {args.out}")
    for entry in report["candidates"]:
        if not entry["exists"]:
            continue
        nonzero = {k: v for k, v in entry["file_counts"].items() if v}
        print(entry["path"], nonzero)


if __name__ == "__main__":
    main()
