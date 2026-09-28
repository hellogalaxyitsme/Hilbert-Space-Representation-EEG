#!/usr/bin/env python3
"""Aggregate BCI IV 2a subject-level ODI dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/bci2a_subject_odi")
    parser.add_argument("--out-dir", default="reports/bci2a_subject_odi_stats")
    parser.add_argument("--modes", default="true")
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    vx = [x - mx for x in xs]
    vy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in vx) * sum(y * y for y in vy))
    if den <= 1e-12:
        return float("nan")
    return sum(x * y for x, y in zip(vx, vy)) / den


def rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[order[j]] == vals[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson(rankdata(xs), rankdata(ys))


def fmt(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.3f}"


def read_rows(reports_dir: Path, modes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*_subject_odi_dynamics.json")):
        payload = json.loads(path.read_text())
        arch = payload["model"]["arch"]
        mode = payload["run"]["label_mode"]
        if mode not in modes:
            continue
        seed = int(payload["run"]["seed"])
        for audit in payload.get("audits", []):
            epoch = int(audit["epoch"])
            for subject in audit.get("subjects", []):
                for layer, metrics in subject.get("layers", {}).items():
                    rows.append(
                        {
                            "dataset": "bci2a",
                            "arch": arch,
                            "mode": mode,
                            "seed": seed,
                            "epoch": epoch,
                            "subject": subject["subject"],
                            "session": subject["session"],
                            "split": subject["split"],
                            "accuracy": subject["accuracy"],
                            "loss": subject["loss"],
                            "n_epochs": subject["n_epochs"],
                            "layer": layer,
                            "input_odi": metrics["input_mean_abs_offdiag_cosine_mean"],
                            "odi": metrics["output_mean_abs_offdiag_cosine_mean"],
                            "odi_std": metrics["output_mean_abs_offdiag_cosine_std"],
                            "collapse_factor": metrics["collapse_factor_mean"],
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cross_subject_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arch"], row["mode"], int(row["seed"]), int(row["epoch"]), row["layer"])].append(row)
    out = []
    for (arch, mode, seed, epoch, layer), vals in grouped.items():
        xs = [float(v["odi"]) for v in vals]
        ys = [float(v["accuracy"]) for v in vals]
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "seed": seed,
                "epoch": epoch,
                "layer": layer,
                "n_subjects": len(vals),
                "pearson_odi_accuracy": pearson(xs, ys),
                "spearman_odi_accuracy": spearman(xs, ys),
                "mean_accuracy": mean(ys),
                "mean_odi": mean(xs),
            }
        )
    return out


def trajectory_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arch"], row["mode"], int(row["seed"]), row["subject"], row["layer"])].append(row)
    out = []
    for (arch, mode, seed, subject, layer), vals in grouped.items():
        vals = sorted(vals, key=lambda row: int(row["epoch"]))
        xs = [float(v["odi"]) for v in vals]
        ys = [float(v["accuracy"]) for v in vals]
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "seed": seed,
                "subject": subject,
                "layer": layer,
                "n_checkpoints": len(vals),
                "pearson_odi_accuracy": pearson(xs, ys),
                "spearman_odi_accuracy": spearman(xs, ys),
                "start_accuracy": ys[0],
                "end_accuracy": ys[-1],
                "delta_accuracy": ys[-1] - ys[0],
                "start_odi": xs[0],
                "end_odi": xs[-1],
                "delta_odi": xs[-1] - xs[0],
            }
        )
    return out


def summarize_cross(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arch"], row["mode"], row["layer"])].append(row)
    out = []
    for (arch, mode, layer), vals in grouped.items():
        ps = [float(v["pearson_odi_accuracy"]) for v in vals if parse_float(v["pearson_odi_accuracy"]) is not None]
        ss = [float(v["spearman_odi_accuracy"]) for v in vals if parse_float(v["spearman_odi_accuracy"]) is not None]
        if not ps:
            continue
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "layer": layer,
                "n_cross_subject_tests": len(ps),
                "mean_pearson_odi_accuracy": mean(ps),
                "mean_abs_pearson_odi_accuracy": mean(abs(v) for v in ps),
                "positive_fraction_pearson": mean(1.0 if v > 0 else 0.0 for v in ps),
                "mean_spearman_odi_accuracy": mean(ss),
                "mean_abs_spearman_odi_accuracy": mean(abs(v) for v in ss),
            }
        )
    return sorted(out, key=lambda row: row["mean_abs_pearson_odi_accuracy"], reverse=True)


def subject_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["arch"], row["mode"], int(row["seed"]), row["subject"], int(row["epoch"]))
        if key in seen:
            continue
        seen.add(key)
        grouped[(row["arch"], row["mode"], int(row["seed"]), row["subject"])].append(row)
    out = []
    for (arch, mode, seed, subject), vals in grouped.items():
        vals = sorted(vals, key=lambda row: int(row["epoch"]))
        acc = [float(v["accuracy"]) for v in vals]
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "seed": seed,
                "subject": subject,
                "split": vals[0]["split"],
                "start_accuracy": acc[0],
                "end_accuracy": acc[-1],
                "best_accuracy": max(acc),
                "delta_accuracy": acc[-1] - acc[0],
            }
        )
    return out


def subject_epoch_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arch"], row["mode"], row["layer"], row["subject"], int(row["epoch"]))].append(row)
    out = []
    for (arch, mode, layer, subject, epoch), vals in grouped.items():
        acc_seen = {}
        for row in vals:
            acc_seen[(row["seed"], row["subject"], row["epoch"])] = float(row["accuracy"])
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "layer": layer,
                "subject": subject,
                "split": vals[0]["split"],
                "epoch": epoch,
                "mean_accuracy": mean(acc_seen.values()),
                "mean_odi": mean(float(v["odi"]) for v in vals),
                "std_odi": std(float(v["odi"]) for v in vals),
                "n_seed_layer_rows": len(vals),
            }
        )
    return sorted(out, key=lambda row: (row["arch"], row["layer"], row["subject"], row["epoch"]))


def summarize_trajectories(traj: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in traj:
        grouped[(row["arch"], row["mode"], row["layer"])].append(row)
    out = []
    for (arch, mode, layer), vals in grouped.items():
        ps = [float(v["pearson_odi_accuracy"]) for v in vals if parse_float(v["pearson_odi_accuracy"]) is not None]
        ss = [float(v["spearman_odi_accuracy"]) for v in vals if parse_float(v["spearman_odi_accuracy"]) is not None]
        if not ps:
            continue
        out.append(
            {
                "arch": arch,
                "mode": mode,
                "layer": layer,
                "n_subject_seed_trajectories": len(ps),
                "mean_pearson_odi_accuracy": mean(ps),
                "mean_abs_pearson_odi_accuracy": mean(abs(v) for v in ps),
                "positive_fraction_pearson": mean(1.0 if v > 0 else 0.0 for v in ps),
                "mean_spearman_odi_accuracy": mean(ss),
                "mean_abs_spearman_odi_accuracy": mean(abs(v) for v in ss),
            }
        )
    return sorted(out, key=lambda row: row["mean_abs_pearson_odi_accuracy"], reverse=True)


def train_heldout_gap(subj: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"train": [], "heldout": []})
    for row in subj:
        if row["mode"] != "true":
            continue
        bucket = "heldout" if row["split"] == "heldout" else "train"
        grouped[row["arch"]][bucket].append(float(row["best_accuracy"]))
    out = []
    for arch, vals in grouped.items():
        train = mean(vals["train"])
        heldout = mean(vals["heldout"])
        out.append(
            {
                "arch": arch,
                "train_subject_best_accuracy": train,
                "heldout_a09_best_accuracy": heldout,
                "train_minus_heldout_gap": train - heldout,
                "n_train_subject_seed_pairs": len(vals["train"]),
                "n_heldout_seed_pairs": len(vals["heldout"]),
            }
        )
    return sorted(out, key=lambda row: row["train_minus_heldout_gap"], reverse=True)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    cross_summary: list[dict[str, Any]],
    traj_summary: list[dict[str, Any]],
    subj: list[dict[str, Any]],
    gap: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    true_subj = [row for row in subj if row["mode"] == "true"]
    by_arch = defaultdict(list)
    for row in true_subj:
        by_arch[row["arch"]].append(float(row["best_accuracy"]))
    lines = [
        "# BCI IV 2a Subject-Level ODI Dynamics",
        "",
        f"Flattened subject/layer/checkpoint rows: {len(rows)}.",
        "",
        "The analysis evaluates each saved dynamics checkpoint on each labeled `AxxT` subject session, then recomputes layer ODI using only that subject's epochs. `A09T` remains marked as the held-out subject from the original training protocol; `A01T`-`A08T` are train-subject sessions.",
        "",
        "Full per-subject ODI dynamics are in `subject_layer_checkpoint_rows.csv`; seed-averaged subject/epoch trajectories are in `subject_epoch_odi_summary.csv`.",
        "",
        "## Per-Architecture Subject Accuracy",
        "",
        "| Architecture | Mean subject best acc | Std | N subject-seed pairs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arch in sorted(by_arch):
        vals = by_arch[arch]
        lines.append(f"| {arch} | {fmt(mean(vals))} | {fmt(std(vals))} | {len(vals)} |")

    lines.extend(
        [
            "",
            "## Train-Subject vs Held-Out A09 Gap",
            "",
            "| Architecture | Train-subject best acc | Held-out A09 best acc | Gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in gap:
        lines.append(
            f"| {row['arch']} | {fmt(row['train_subject_best_accuracy'])} | "
            f"{fmt(row['heldout_a09_best_accuracy'])} | {fmt(row['train_minus_heldout_gap'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Cross-Subject ODI/Accuracy Links",
            "",
            "Rows average cross-subject ODI-vs-accuracy correlations over seeds and checkpoints.",
            "",
            "| Architecture | Layer | Tests | Mean Pearson | Mean |r| | Positive frac | Mean Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [r for r in cross_summary if r["mode"] == "true"][:20]:
        lines.append(
            f"| {row['arch']} | `{row['layer']}` | {row['n_cross_subject_tests']} | "
            f"{fmt(row['mean_pearson_odi_accuracy'])} | {fmt(row['mean_abs_pearson_odi_accuracy'])} | "
            f"{fmt(row['positive_fraction_pearson'])} | {fmt(row['mean_spearman_odi_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Within-Subject ODI/Accuracy Dynamics",
            "",
            "Rows average ODI-vs-accuracy correlations over each subject/seed trajectory across checkpoints.",
            "",
            "| Architecture | Layer | Subject-seed trajectories | Mean Pearson | Mean |r| | Positive frac | Mean Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [r for r in traj_summary if r["mode"] == "true"][:20]:
        lines.append(
            f"| {row['arch']} | `{row['layer']}` | {row['n_subject_seed_trajectories']} | "
            f"{fmt(row['mean_pearson_odi_accuracy'])} | {fmt(row['mean_abs_pearson_odi_accuracy'])} | "
            f"{fmt(row['positive_fraction_pearson'])} | {fmt(row['mean_spearman_odi_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This subject-level utility and diagnostic analysis uses the existing training results.",
            "- Positive ODI/accuracy correlation means subjects with larger layer-wise band-subspace mixing at that checkpoint tend to have higher accuracy.",
            "- Negative correlation means lower ODI, i.e. stronger preservation/separation of frequency-band directions, is associated with better subject accuracy.",
            "- Subjects `A01T`-`A08T` contributed to training and `A09T` is held out; these cross-subject correlations quantify heterogeneity within this split. Separate held-out evaluations estimate generalization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.reports_dir), set(args.modes.split()))
    cross = cross_subject_correlations(rows)
    traj = trajectory_correlations(rows)
    cross_summary = summarize_cross(cross)
    traj_summary = summarize_trajectories(traj)
    subj = subject_summary(rows)
    subj_epoch = subject_epoch_summary(rows)
    gap = train_heldout_gap(subj)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "subject_layer_checkpoint_rows.csv", rows)
    write_csv(out_dir / "cross_subject_odi_accuracy_correlations.csv", cross)
    write_csv(out_dir / "subject_odi_accuracy_trajectory_correlations.csv", traj)
    write_csv(out_dir / "cross_subject_odi_accuracy_summary.csv", cross_summary)
    write_csv(out_dir / "subject_accuracy_summary.csv", subj)
    write_csv(out_dir / "subject_epoch_odi_summary.csv", subj_epoch)
    write_csv(out_dir / "subject_odi_accuracy_trajectory_summary.csv", traj_summary)
    write_csv(out_dir / "train_heldout_subject_gap.csv", gap)
    write_markdown(out_dir / "bci2a_subject_odi_summary.md", rows, cross_summary, traj_summary, subj, gap)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
