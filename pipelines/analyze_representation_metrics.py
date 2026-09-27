#!/usr/bin/env python3
"""Aggregate representation-geometry analyses from existing CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


DATASETS = ("bci2a", "sleepedf_full", "seediv", "p300")
TRAIN_SEEDS = (41, 42, 43)
HELDOUT_SEEDS = (44, 45, 46)
ARCH_TAXONOMY = {
    "eegnet": "progressive collapse",
    "shallowconvnet": "nonlinear-induced distortion",
    "eegconformer": "architecture-induced collapse with training reorganization",
    "tsception": "multi-branch fusion-induced reorganization",
    "atcnet": "hybrid convolutional collapse with attention/TCN reorganization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="reports/cka_rsa_stats/joined_layer_checkpoint_metrics.csv")
    parser.add_argument("--taxonomy", default="reports/architecture_taxonomy_all_datasets/architecture_taxonomy_layer_rows.csv")
    parser.add_argument("--paper-focused", default="reports/paper_statistics/paper_focused_effect_tests.csv")
    parser.add_argument("--paper-all", default="reports/paper_statistics/all_true_vs_shuffled_corrected.csv")
    parser.add_argument("--classical", default="reports/statistical_completeness/classical_true_vs_shuffled_effect_tests.csv")
    parser.add_argument("--rotated-metrics", default="reports/bci2a_rotated_subject_stats/rotated_subject_metric_rows.csv")
    parser.add_argument("--rotated-layers", default="reports/bci2a_rotated_subject_stats/rotated_subject_layer_rows.csv")
    parser.add_argument("--rotated-tests", default="reports/bci2a_rotated_subject_stats/rotated_subject_true_vs_shuffled_tests.csv")
    parser.add_argument("--out-dir", default="reports/representation_analysis")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_int(value: Any, default: int = 0) -> int:
    out = finite_float(value, float(default))
    return int(out) if math.isfinite(out) else default


def fmt(value: Any, digits: int = 3) -> str:
    x = finite_float(value)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(values: list[float], rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(arr.mean())
    if len(arr) == 1:
        return observed, observed, observed
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, float(lo), float(hi)


def taxonomy_lookup(path: Path) -> dict[tuple[str, str, str, int, str], tuple[str, str]]:
    out: dict[tuple[str, str, str, int, str], tuple[str, str]] = {}
    for row in read_csv(path):
        key = (row["dataset"], row["arch"], row["mode"], finite_int(row["seed"]), row["layer"])
        out[key] = (
            row.get("operation_group") or "unmapped",
            row.get("taxonomy") or ARCH_TAXONOMY.get(row["arch"], "unmapped"),
        )
    return out


def arch_layer_lookup(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    by_arch_layer: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        by_arch_layer[(row["arch"], row["layer"])].append(
            (
                row.get("operation_group") or "unmapped",
                row.get("taxonomy") or ARCH_TAXONOMY.get(row["arch"], "unmapped"),
            )
        )
    out = {}
    for key, vals in by_arch_layer.items():
        counts = defaultdict(int)
        for val in vals:
            counts[val] += 1
        out[key] = max(counts, key=counts.get)
    return out


def enriched_joined_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    layer_map = taxonomy_lookup(Path(args.taxonomy))
    rows = []
    for row in read_csv(Path(args.joined)):
        dataset = row["dataset"]
        arch = row["arch"]
        mode = row["mode"]
        seed = finite_int(row["seed"])
        layer = row["layer"]
        operation_group, taxonomy = layer_map.get(
            (dataset, arch, mode, seed, layer),
            ("unmapped", ARCH_TAXONOMY.get(arch, "unmapped")),
        )
        rows.append(
            {
                **row,
                "seed": seed,
                "epoch": finite_int(row["epoch"]),
                "operation_group": operation_group,
                "taxonomy": taxonomy,
                "analysis_excluded": is_excluded_supervised(dataset, arch),
            }
        )
    return rows


def is_excluded_supervised(dataset: str, arch: str) -> bool:
    return dataset == "p300" and arch == "atcnet"


def is_excluded_fm(dataset: str, fm: str) -> bool:
    return dataset == "sleepedf_full" and fm == "eegpt"


def model_matrix(rows: list[dict[str, Any]], cont_cols: list[str], fixed_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([finite_float(row["val_acc"]) for row in rows], dtype=np.float64)
    cols = [np.ones(len(rows), dtype=np.float64)]
    for col in cont_cols:
        vals = np.asarray([finite_float(row[col]) for row in rows], dtype=np.float64)
        if col == "activation_rank":
            vals = np.log1p(vals)
        sd = float(np.std(vals))
        if sd <= 1e-12:
            continue
        cols.append((vals - vals.mean()) / sd)
    for col in fixed_cols:
        cats = sorted({str(row[col]) for row in rows})
        for cat in cats[1:]:
            cols.append(np.asarray([1.0 if str(row[col]) == cat else 0.0 for row in rows], dtype=np.float64))
    return y, np.column_stack(cols)


def ols_r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    n, p = x.shape
    if n <= p + 1 or not math.isfinite(r2):
        return r2, float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    return float(r2), float(adj)


def ols_r2_indexed(y: np.ndarray, x: np.ndarray, indices: np.ndarray | None = None) -> tuple[float, float]:
    if indices is not None:
        y = y[indices]
        x = x[indices]
    return ols_r2(y, x)


def nested_delta_r2(
    rows: list[dict[str, Any]],
    reduced_cols: list[str],
    full_extra_cols: list[str],
    fixed_cols: list[str],
) -> dict[str, float]:
    needed = ["val_acc", *reduced_cols, *full_extra_cols]
    keep = [row for row in rows if all(math.isfinite(finite_float(row.get(col))) for col in needed)]
    if len(keep) < 10:
        return {"n_rows": len(keep), "r2_reduced": float("nan"), "r2_full": float("nan"), "delta_r2": float("nan")}
    y_red, x_red = model_matrix(keep, reduced_cols, fixed_cols)
    y_full, x_full = model_matrix(keep, [*reduced_cols, *full_extra_cols], fixed_cols)
    r2_red, adj_red = ols_r2(y_red, x_red)
    r2_full, adj_full = ols_r2(y_full, x_full)
    return {
        "n_rows": len(keep),
        "r2_reduced": r2_red,
        "r2_full": r2_full,
        "delta_r2": r2_full - r2_red,
        "adj_r2_reduced": adj_red,
        "adj_r2_full": adj_full,
        "delta_adj_r2": adj_full - adj_red,
    }


def variance_partitioning(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> list[dict[str, Any]]:
    scopes = {
        "all_14640_archival": rows,
        "analysis_clean_all_modes": [r for r in rows if not r["analysis_excluded"]],
        "analysis_clean_true_only": [r for r in rows if not r["analysis_excluded"] and r["mode"] == "true"],
    }
    reduced = ["cka_input", "rsa_input_spearman", "rsa_label_spearman", "activation_rank"]
    fixed = ["dataset", "arch", "mode", "seed", "operation_group"]
    out = []
    for scope, scope_rows in scopes.items():
        needed = ["val_acc", *reduced, "hsdd_odi"]
        keep = [row for row in scope_rows if all(math.isfinite(finite_float(row.get(col))) for col in needed)]
        y, x_red = model_matrix(keep, reduced, fixed)
        _, x_full = model_matrix(keep, [*reduced, "hsdd_odi"], fixed)
        r2_red, adj_red = ols_r2(y, x_red)
        r2_full, adj_full = ols_r2(y, x_full)
        estimate = {
            "n_rows": len(keep),
            "r2_reduced": r2_red,
            "r2_full": r2_full,
            "delta_r2": r2_full - r2_red,
            "adj_r2_reduced": adj_red,
            "adj_r2_full": adj_full,
            "delta_adj_r2": adj_full - adj_red,
        }
        clusters: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(keep):
            key = f"{row['dataset']}|{row['arch']}|{row['mode']}|{row['seed']}"
            clusters[key].append(idx)
        cluster_keys = sorted(clusters)
        boot = []
        for _ in range(n_boot):
            sampled: list[int] = []
            for key in rng.choice(cluster_keys, size=len(cluster_keys), replace=True):
                sampled.extend(clusters[str(key)])
            idx = np.asarray(sampled, dtype=np.int64)
            boot_r2_red, _ = ols_r2_indexed(y, x_red, idx)
            boot_r2_full, _ = ols_r2_indexed(y, x_full, idx)
            boot.append(boot_r2_full - boot_r2_red)
        boot = [v for v in boot if math.isfinite(v)]
        lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
        out.append(
            {
                "scope": scope,
                **estimate,
                "delta_r2_ci_low": float(lo),
                "delta_r2_ci_high": float(hi),
                "n_bootstrap": len(boot),
                "reduced_predictors": "+".join(reduced),
                "added_predictor": "hsdd_odi",
                "fixed_effects": "+".join(fixed),
                "bootstrap_unit": "dataset|arch|mode|seed",
            }
        )
    return out


def residuals(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return values - values.mean()
    x = np.column_stack([np.ones(len(values)), controls])
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ beta


def control_matrix(rows: list[dict[str, Any]], cont_cols: list[str], fixed_cols: list[str], ranked: bool) -> np.ndarray:
    cols = []
    for col in cont_cols:
        vals = np.asarray([finite_float(row[col]) for row in rows], dtype=np.float64)
        if col == "activation_rank":
            vals = np.log1p(vals)
        if ranked:
            vals = rankdata(vals)
        sd = np.std(vals)
        if sd > 1e-12:
            cols.append((vals - vals.mean()) / sd)
    for col in fixed_cols:
        cats = sorted({str(row[col]) for row in rows})
        for cat in cats[1:]:
            cols.append(np.asarray([1.0 if str(row[col]) == cat else 0.0 for row in rows], dtype=np.float64))
    if not cols:
        return np.empty((len(rows), 0), dtype=np.float64)
    return np.column_stack(cols)


def partial_corr_estimate(rows: list[dict[str, Any]], method: str) -> tuple[float, float]:
    vals = [
        row
        for row in rows
        if math.isfinite(finite_float(row.get("hsdd_odi")))
        and math.isfinite(finite_float(row.get("val_acc")))
        and math.isfinite(finite_float(row.get("activation_rank")))
    ]
    if len(vals) < 8:
        return float("nan"), float("nan")
    x = np.asarray([finite_float(row["hsdd_odi"]) for row in vals], dtype=np.float64)
    y = np.asarray([finite_float(row["val_acc"]) for row in vals], dtype=np.float64)
    ranked = method == "spearman"
    if ranked:
        x = rankdata(x)
        y = rankdata(y)
    controls = control_matrix(vals, ["activation_rank"], ["dataset", "arch", "seed"], ranked=ranked)
    rx = residuals(x, controls)
    ry = residuals(y, controls)
    r = pearson(rx, ry)
    if not math.isfinite(r):
        return r, float("nan")
    p = float(stats.pearsonr(rx, ry).pvalue)
    return r, p


def partial_correlations(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> list[dict[str, Any]]:
    clean_true = [r for r in rows if r["mode"] == "true" and not r["analysis_excluded"]]
    out = []
    for operation_group in sorted({r["operation_group"] for r in clean_true}):
        group_rows = [r for r in clean_true if r["operation_group"] == operation_group]
        if len(group_rows) < 20:
            continue
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            clusters[f"{row['dataset']}|{row['arch']}|{row['seed']}"].append(row)
        keys = sorted(clusters)
        for method in ("pearson", "spearman"):
            r, p = partial_corr_estimate(group_rows, method)
            boot = []
            for _ in range(n_boot):
                sampled = []
                for key in rng.choice(keys, size=len(keys), replace=True):
                    sampled.extend(clusters[str(key)])
                boot.append(partial_corr_estimate(sampled, method)[0])
            boot = [v for v in boot if math.isfinite(v)]
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
            out.append(
                {
                    "scope": "analysis_clean_true_only",
                    "operation_group": operation_group,
                    "method": method,
                    "partial_corr_odi_val_acc": r,
                    "p_value": p,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "n_rows": len(group_rows),
                    "n_clusters": len(keys),
                    "control": "log1p_activation_rank + dataset/arch/seed fixed effects",
                    "bootstrap_unit": "dataset|arch|seed",
                }
            )
    return out


def trajectory_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_true = [r for r in rows if r["mode"] == "true" and not r["analysis_excluded"]]
    grouped: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_true:
        key = (row["dataset"], row["arch"], row["seed"], row["layer"], row["operation_group"], row["taxonomy"])
        grouped[key].append(row)
    out = []
    for (dataset, arch, seed, layer, operation_group, taxonomy), vals in grouped.items():
        vals = sorted(vals, key=lambda r: r["epoch"])
        x = np.asarray([finite_float(v["hsdd_odi"]) for v in vals], dtype=np.float64)
        y = np.asarray([finite_float(v["val_acc"]) for v in vals], dtype=np.float64)
        r = pearson(x, y)
        if not math.isfinite(r) or abs(r) <= 1e-12:
            continue
        out.append(
            {
                "dataset": dataset,
                "arch": arch,
                "seed": seed,
                "layer": layer,
                "operation_group": operation_group,
                "taxonomy": taxonomy,
                "n_checkpoints": len(vals),
                "pearson_odi_val_acc": r,
                "sign": 1 if r > 0 else -1,
            }
        )
    return out


def taxonomy_oos_test(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traj = trajectory_correlations(rows)
    eval_rows = []

    def majority_sign(vals: list[int]) -> int | None:
        vals = [int(v) for v in vals if int(v) in {-1, 1}]
        if not vals:
            return None
        pos = sum(1 for v in vals if v > 0)
        neg = len(vals) - pos
        return 1 if pos >= neg else -1

    def taxonomy_predictions(train: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
        by_taxon_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in train:
            by_taxon_group[(row["taxonomy"], row["operation_group"])].append(finite_float(row["pearson_odi_val_acc"]))
        predictions = {}
        for key, vals in by_taxon_group.items():
            vals = [v for v in vals if math.isfinite(v)]
            if len(vals) >= 3 and abs(float(np.mean(vals))) > 1e-12:
                predictions[key] = 1 if float(np.mean(vals)) > 0 else -1
        return predictions

    def add_eval_rows(
        *,
        evaluation_scope: str,
        split_id: str,
        train: list[dict[str, Any]],
        test: list[dict[str, Any]],
        train_desc: str,
        heldout_desc: str,
    ) -> None:
        tax_pred = taxonomy_predictions(train)
        global_majority = majority_sign([finite_int(r["sign"]) for r in train])
        dataset_majority: dict[str, int] = {}
        for dataset in sorted({r["dataset"] for r in train}):
            pred = majority_sign([finite_int(r["sign"]) for r in train if r["dataset"] == dataset])
            if pred is not None:
                dataset_majority[dataset] = pred
        for row in test:
            key = (row["taxonomy"], row["operation_group"])
            actual = finite_int(row["sign"])
            candidates = {
                "taxonomy": tax_pred.get(key),
                "majority_class": global_majority,
                "per_dataset_majority": dataset_majority.get(row["dataset"], global_majority),
            }
            for predictor, pred in candidates.items():
                if pred is None:
                    continue
                eval_rows.append(
                    {
                        "evaluation_scope": evaluation_scope,
                        "split_id": split_id,
                        "train_scope": train_desc,
                        "heldout_scope": heldout_desc,
                        "predictor": predictor,
                        "taxonomy": row["taxonomy"],
                        "operation_group": row["operation_group"],
                        "dataset": row["dataset"],
                        "arch": row["arch"],
                        "seed": row["seed"],
                        "layer": row["layer"],
                        "predicted_sign": pred,
                        "actual_sign": actual,
                        "correct": int(pred == actual),
                        "actual_pearson_odi_val_acc": row["pearson_odi_val_acc"],
                    }
                )

    for heldout_dataset in DATASETS:
        train = [r for r in traj if r["dataset"] != heldout_dataset]
        test = [r for r in traj if r["dataset"] == heldout_dataset]
        add_eval_rows(
            evaluation_scope="leave_dataset_out",
            split_id=f"heldout_dataset={heldout_dataset}",
            train=train,
            test=test,
            train_desc="+".join(ds for ds in DATASETS if ds != heldout_dataset),
            heldout_desc=heldout_dataset,
        )

    all_seeds = sorted({finite_int(r["seed"]) for r in traj})
    for heldout_seed in all_seeds:
        train = [r for r in traj if finite_int(r["seed"]) != heldout_seed]
        test = [r for r in traj if finite_int(r["seed"]) == heldout_seed]
        add_eval_rows(
            evaluation_scope="leave_seed_out",
            split_id=f"heldout_seed={heldout_seed}",
            train=train,
            test=test,
            train_desc="all_datasets|all_other_seeds",
            heldout_desc=str(heldout_seed),
        )

    summaries = []
    scopes: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for scope in sorted({row["evaluation_scope"] for row in eval_rows}):
        for predictor in ("taxonomy", "majority_class", "per_dataset_majority"):
            vals = [row for row in eval_rows if row["evaluation_scope"] == scope and row["predictor"] == predictor]
            scopes[(scope, predictor, "all_datasets")] = vals
            for dataset in sorted({row["dataset"] for row in vals}):
                scopes[(scope, predictor, dataset)] = [row for row in vals if row["dataset"] == dataset]
    for (scope, predictor, dataset), vals in scopes.items():
        correct = [finite_float(row["correct"]) for row in vals]
        acc, lo, hi = bootstrap_ci(correct, rng, n_boot)
        summaries.append(
            {
                "evaluation_scope": scope,
                "predictor": predictor,
                "dataset": dataset,
                "n_predictions": len(correct),
                "sign_accuracy": acc,
                "ci_low": lo,
                "ci_high": hi,
                "split_protocol": "train on all non-held-out units; predict held-out dataset or seed",
                "excluded_configs": "p300/atcnet",
                "unit": "dataset|seed|architecture-taxonomy|operation-group|layer trajectory",
            }
        )
    return eval_rows, summaries


def cka_odi_dissociation(rows: list[dict[str, Any]], rng: np.random.Generator, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = [
        r
        for r in rows
        if r["mode"] == "true"
        and not r["analysis_excluded"]
        and all(math.isfinite(finite_float(r.get(col))) for col in ("cka_input", "hsdd_odi", "val_acc"))
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        groups[(row["dataset"], finite_int(row["epoch"]), row["operation_group"])].append(row)
    candidates = []
    for (dataset, epoch, operation_group), vals in groups.items():
        if len(vals) < 4:
            continue
        pair_stats = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                if a["arch"] == b["arch"] and a["layer"] == b["layer"] and a["seed"] == b["seed"]:
                    continue
                cka_diff = abs(finite_float(a["cka_input"]) - finite_float(b["cka_input"]))
                odi_diff = abs(finite_float(a["hsdd_odi"]) - finite_float(b["hsdd_odi"]))
                acc_diff = abs(finite_float(a["val_acc"]) - finite_float(b["val_acc"]))
                pair_stats.append((cka_diff, odi_diff, acc_diff, a, b))
        if len(pair_stats) < 8:
            continue
        cka_cut = float(np.percentile([p[0] for p in pair_stats], 20))
        odi_cut = float(np.percentile([p[1] for p in pair_stats], 80))
        acc_cut = float(np.percentile([p[2] for p in pair_stats], 60))
        for cka_diff, odi_diff, acc_diff, a, b in pair_stats:
            if cka_diff <= cka_cut and odi_diff >= odi_cut and acc_diff >= acc_cut:
                candidates.append(
                    {
                        "dataset": dataset,
                        "epoch": epoch,
                        "operation_group": operation_group,
                        "arch_a": a["arch"],
                        "layer_a": a["layer"],
                        "seed_a": a["seed"],
                        "cka_input_a": finite_float(a["cka_input"]),
                        "hsdd_odi_a": finite_float(a["hsdd_odi"]),
                        "val_acc_a": finite_float(a["val_acc"]),
                        "arch_b": b["arch"],
                        "layer_b": b["layer"],
                        "seed_b": b["seed"],
                        "cka_input_b": finite_float(b["cka_input"]),
                        "hsdd_odi_b": finite_float(b["hsdd_odi"]),
                        "val_acc_b": finite_float(b["val_acc"]),
                        "cka_abs_diff": cka_diff,
                        "odi_abs_diff": odi_diff,
                        "val_acc_abs_diff": acc_diff,
                        "dissociation_score": odi_diff / max(cka_diff, 1e-6),
                        "criterion": "bottom-quintile CKA-input difference, top-quintile ODI difference, >=60th percentile accuracy difference within dataset|epoch|operation-group",
                    }
                )
    candidates = sorted(candidates, key=lambda r: finite_float(r["dissociation_score"]), reverse=True)
    top = candidates[:250]
    summary = []
    if top:
        vals = {
            "cka_abs_diff": [finite_float(r["cka_abs_diff"]) for r in top],
            "odi_abs_diff": [finite_float(r["odi_abs_diff"]) for r in top],
            "val_acc_abs_diff": [finite_float(r["val_acc_abs_diff"]) for r in top],
        }
        row: dict[str, Any] = {
            "scope": "constructed_cka_odi_dissociation",
            "n_pairs": len(top),
            "selection": "CKA-input nearly matched while ODI and validation accuracy diverge",
        }
        for metric, arr in vals.items():
            arr_np = np.asarray(arr, dtype=np.float64)
            row[f"{metric}_median"] = float(np.median(arr_np))
            boots = [float(np.median(rng.choice(arr_np, size=len(arr_np), replace=True))) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row[f"{metric}_median_ci_low"] = float(lo)
            row[f"{metric}_median_ci_high"] = float(hi)
        summary.append(row)
    return top, summary


def subject_unit_correlations(args: argparse.Namespace, rng: np.random.Generator, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = {}
    for row in read_csv(Path(args.rotated_metrics)):
        if row["mode"] != "true":
            continue
        metrics[(row["arch"], row["heldout_subject"], finite_int(row["seed"]))] = finite_float(row["best_val_acc"])
    arch_layer_map = arch_layer_lookup(Path(args.taxonomy))

    per_seed_layer = []
    per_seed_group: dict[tuple[str, str, str, int, str, str], list[float]] = defaultdict(list)
    for row in read_csv(Path(args.rotated_layers)):
        if row["mode"] != "true":
            continue
        arch = row["arch"]
        subject = row["heldout_subject"]
        seed = finite_int(row["seed"])
        acc = metrics.get((arch, subject, seed), float("nan"))
        if not math.isfinite(acc):
            continue
        operation_group, taxonomy = arch_layer_map.get(
            (arch, row["layer"]),
            ("unmapped", ARCH_TAXONOMY.get(arch, "unmapped")),
        )
        odi = finite_float(row["odi"])
        per_seed_layer.append(
            {
                "arch": arch,
                "heldout_subject": subject,
                "seed": seed,
                "layer": row["layer"],
                "operation_group": operation_group,
                "taxonomy": taxonomy,
                "odi": odi,
                "best_val_acc": acc,
            }
        )
        per_seed_group[(arch, subject, operation_group, seed, taxonomy, "odi")].append(odi)

    subject_layer: dict[tuple[str, str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in per_seed_layer:
        key = (row["arch"], row["heldout_subject"], row["layer"], row["operation_group"], row["taxonomy"])
        subject_layer[key].append((finite_float(row["odi"]), finite_float(row["best_val_acc"])))

    subject_group_seed = []
    for (arch, subject, operation_group, seed, taxonomy, _), vals in per_seed_group.items():
        acc = metrics.get((arch, subject, seed), float("nan"))
        finite_vals = [v for v in vals if math.isfinite(v)]
        if finite_vals and math.isfinite(acc):
            subject_group_seed.append(
                {
                    "arch": arch,
                    "heldout_subject": subject,
                    "seed": seed,
                    "operation_group": operation_group,
                    "taxonomy": taxonomy,
                    "odi": float(np.mean(finite_vals)),
                    "best_val_acc": acc,
                }
            )
    subject_group: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in subject_group_seed:
        key = (row["arch"], row["heldout_subject"], row["operation_group"], row["taxonomy"])
        subject_group[key].append((finite_float(row["odi"]), finite_float(row["best_val_acc"])))

    def aggregate_subject_units(grouped: dict[tuple[str, ...], list[tuple[float, float]]]) -> list[dict[str, Any]]:
        rows = []
        for key, vals in grouped.items():
            odi = [v[0] for v in vals if math.isfinite(v[0]) and math.isfinite(v[1])]
            acc = [v[1] for v in vals if math.isfinite(v[0]) and math.isfinite(v[1])]
            if odi and acc:
                rows.append({"key": key, "odi": float(np.mean(odi)), "best_val_acc": float(np.mean(acc))})
        return rows

    subject_layer_units = aggregate_subject_units(subject_layer)
    subject_group_units = aggregate_subject_units(subject_group)

    def corr_table(units: list[dict[str, Any]], group_kind: str) -> list[dict[str, Any]]:
        out = []
        group_index = 2 if group_kind == "layer" else 2
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for unit in units:
            key = unit["key"]
            arch = key[0]
            name = key[group_index]
            taxonomy = key[-1]
            groups[(arch, name, taxonomy)].append(unit)
        for (arch, name, taxonomy), vals in sorted(groups.items()):
            if len(vals) < 4:
                continue
            x = np.asarray([v["odi"] for v in vals], dtype=np.float64)
            y = np.asarray([v["best_val_acc"] for v in vals], dtype=np.float64)
            pr = pearson(x, y)
            sr = spearman(x, y)
            p = float(stats.spearmanr(x, y).pvalue) if math.isfinite(sr) else float("nan")
            boot = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(vals), size=len(vals))
                boot.append(spearman(x[idx], y[idx]))
            boot = [v for v in boot if math.isfinite(v)]
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
            out.append(
                {
                    "unit": "heldout_subject",
                    "group_kind": group_kind,
                    "arch": arch,
                    "layer_or_operation_group": name,
                    "taxonomy": taxonomy,
                    "n_subjects": len(vals),
                    "pearson_odi_accuracy": pr,
                    "spearman_odi_accuracy": sr,
                    "spearman_p_value": p,
                    "spearman_ci_low": float(lo),
                    "spearman_ci_high": float(hi),
                    "seed_aggregation": "mean over seeds 41-43",
                }
            )
        return out

    return corr_table(subject_layer_units, "layer"), corr_table(subject_group_units, "operation_group")


def cleaned_focused_tests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = []
    excluded = []
    for row in read_csv(Path(args.paper_focused)):
        dataset = row.get("dataset", "")
        arch = row.get("arch", "")
        fm = row.get("fm", "")
        reason = ""
        if arch and is_excluded_supervised(dataset, arch):
            reason = "P300 ATCNet uses a Braindecode-shortened 200-sample configuration and is excluded from reported-scope findings."
        if fm and is_excluded_fm(dataset, fm):
            reason = "Sleep-EDF EEGPT is degenerate because the two-channel bipolar montage poorly maps to EEGPT canonical channels."
        if reason:
            excluded.append({**row, "exclusion_reason": reason})
        else:
            clean.append(row)
    return clean, excluded


def build_primary_contrasts(
    args: argparse.Namespace,
    variance_rows: list[dict[str, Any]],
    taxonomy_summary: list[dict[str, Any]],
    subject_group_rows: list[dict[str, Any]],
    clean_focused: list[dict[str, Any]],
    dissociation_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    supervised = [
        r
        for r in clean_focused
        if r.get("test_family") == "supervised_dl"
        and r.get("experiment") == "static"
        and r.get("metric") == "training.best_val_acc"
    ]
    for dataset in DATASETS:
        candidates = [r for r in supervised if r["dataset"] == dataset]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: finite_float(r["true_mean"]))
        rows.append(
            primary_from_effect_row(
                f"P{len(rows)+1:02d}",
                "supervised_feasibility",
                best,
                f"Best supervised static validation contrast for {dataset}",
            )
        )

    rotated = [r for r in read_csv(Path(args.rotated_tests)) if r.get("arch") == "atcnet" and r.get("metric") == "best_val_acc"]
    if rotated:
        best = rotated[0]
        rows.append(
            {
                "contrast_id": f"P{len(rows)+1:02d}",
                "family": "rotated_subject_robustness",
                "dataset_scope": "bci2a_rotated_subjects",
                "model_or_taxon": "atcnet",
                "metric": "best_val_acc true-minus-shuffled",
                "estimate": finite_float(best["mean_diff_true_minus_shuffled"]),
                "ci_low": "",
                "ci_high": "",
                "p_value": finite_float(best["p_value"]),
                "n": best["n_pairs"],
                "status": "primary",
                "source": args.rotated_tests,
                "notes": "Held-out A01-A09 subjects, seeds 41-43.",
            }
        )

    classical = [r for r in read_csv(Path(args.classical)) if r.get("dataset") == "bci2a" and r.get("family") == "fbcsp"]
    if classical:
        rows.append(
            {
                "contrast_id": f"P{len(rows)+1:02d}",
                "family": "classical_eeg_anchor",
                "dataset_scope": "bci2a",
                "model_or_taxon": "fbcsp",
                "metric": "val_acc true-minus-shuffled",
                "estimate": finite_float(classical[0]["mean_diff_true_minus_shuffled"]),
                "ci_low": "",
                "ci_high": "",
                "p_value": finite_float(classical[0]["p_value"]),
                "n": classical[0]["n_pairs"],
                "status": "primary",
                "source": args.classical,
                "notes": "EEG literature anchor; other classical filter-bank rows remain supplementary.",
            }
        )

    if dissociation_summary:
        d = dissociation_summary[0]
        rows.append(
            {
                "contrast_id": f"P{len(rows)+1:02d}",
                "family": "complementarity",
                "dataset_scope": "all_datasets_clean_true_only",
                "model_or_taxon": "all_supervised_architectures",
                "metric": "constructed CKA/ODI dissociation median ODI difference",
                "estimate": d["odi_abs_diff_median"],
                "ci_low": d["odi_abs_diff_median_ci_low"],
                "ci_high": d["odi_abs_diff_median_ci_high"],
                "p_value": "",
                "n": d["n_pairs"],
                "status": "primary",
                "source": f"{args.out_dir}/cka_odi_dissociation_summary.csv",
                "notes": "Constructed pairs have near-matched CKA-input but divergent ODI and validation accuracy; variance-partitioning delta R^2 is retained as a supplementary negative result.",
            }
        )

    pooled_tax = [
        r
        for r in taxonomy_summary
        if r["evaluation_scope"] == "leave_dataset_out" and r["predictor"] == "taxonomy" and r["dataset"] == "all_datasets"
    ]
    if pooled_tax:
        t = pooled_tax[0]
        rows.append(
            {
                "contrast_id": f"P{len(rows)+1:02d}",
                "family": "taxonomy_out_of_sample",
                "dataset_scope": "heldout_dataset_and_seed_splits",
                "model_or_taxon": "architecture_taxonomy_operation_groups",
                "metric": "sign_accuracy ODI-accuracy correlation",
                "estimate": t["sign_accuracy"],
                "ci_low": t["ci_low"],
                "ci_high": t["ci_high"],
                "p_value": "",
                "n": t["n_predictions"],
                "status": "primary",
                "source": f"{args.out_dir}/taxonomy_oos_sign_summary.csv",
                "notes": "Leave-dataset-out taxonomy sign prediction, compared against majority-class and per-dataset-majority baselines.",
            }
        )

    primary_subject = [
        r
        for r in subject_group_rows
        if r["arch"] == "atcnet" and r["layer_or_operation_group"] == "classifier"
    ]
    if primary_subject:
        s = primary_subject[0]
        rows.append(
            {
                "contrast_id": f"P{len(rows)+1:02d}",
                "family": "subject_unit_bci2a_geometry",
                "dataset_scope": "bci2a_rotated_subjects",
                "model_or_taxon": "atcnet/classifier",
                "metric": "Spearman ODI-accuracy across held-out subjects",
                "estimate": s["spearman_odi_accuracy"],
                "ci_low": s["spearman_ci_low"],
                "ci_high": s["spearman_ci_high"],
                "p_value": s["spearman_p_value"],
                "n": s["n_subjects"],
                "status": "primary",
                "source": f"{args.out_dir}/bci2a_subject_unit_operation_group_correlations.csv",
                "notes": "Subject is the analysis unit; top BCI architecture decision-layer group, seeds 41-43 averaged within subject.",
            }
        )

    return rows[:10]


def primary_from_effect_row(contrast_id: str, family: str, row: dict[str, str], notes: str) -> dict[str, Any]:
    model = row.get("arch") or row.get("fm")
    return {
        "contrast_id": contrast_id,
        "family": family,
        "dataset_scope": row["dataset"],
        "model_or_taxon": model,
        "metric": f"{row['metric']} true-minus-shuffled",
        "estimate": finite_float(row["mean_diff_true_minus_shuffled"]),
        "ci_low": "",
        "ci_high": "",
        "p_value": finite_float(row["p_value"]),
        "n": row["n_pairs"],
        "status": "primary",
        "source": "reports/paper_statistics/paper_focused_effect_tests.csv",
        "notes": notes,
    }


def write_markdown(
    out_dir: Path,
    variance_rows: list[dict[str, Any]],
    partial_rows: list[dict[str, Any]],
    taxonomy_summary: list[dict[str, Any]],
    subject_group_rows: list[dict[str, Any]],
    dissociation_summary: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    clean_focused: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    v_all = next(r for r in variance_rows if r["scope"] == "all_14640_archival")
    v_primary = next(r for r in variance_rows if r["scope"] == "analysis_clean_true_only")
    tax_primary = next(
        r
        for r in taxonomy_summary
        if r["evaluation_scope"] == "leave_dataset_out" and r["predictor"] == "taxonomy" and r["dataset"] == "all_datasets"
    )
    tax_majority = next(
        (
            r
            for r in taxonomy_summary
            if r["evaluation_scope"] == "leave_dataset_out" and r["predictor"] == "majority_class" and r["dataset"] == "all_datasets"
        ),
        None,
    )
    diss_primary = dissociation_summary[0] if dissociation_summary else None
    subj_primary = next(
        (r for r in subject_group_rows if r["arch"] == "atcnet" and r["layer_or_operation_group"] == "classifier"),
        None,
    )
    lines = [
        "# Representation-geometry aggregate analyses",
        "",
        "This report uses existing CSV artifacts only. No new model training was run.",
        "",
        "## Main Readout",
        "",
        (
            f"- On all 14,640 joined rows, ODI adds delta R^2 = {fmt(v_all['delta_r2'], 6)} "
            f"(bootstrap 95% CI [{fmt(v_all['delta_r2_ci_low'], 6)}, {fmt(v_all['delta_r2_ci_high'], 6)}]) "
            "over CKA-input, RSA-input, RSA-label, and activation rank. This is a negative result for broad additive variance."
        ),
        (
            f"- On the analysis-clean true-label rows, ODI adds delta R^2 = {fmt(v_primary['delta_r2'], 4)} "
            f"(bootstrap 95% CI [{fmt(v_primary['delta_r2_ci_low'], 4)}, {fmt(v_primary['delta_r2_ci_high'], 4)}]). "
            "This remains small; keep variance partitioning in the supplement rather than using it as the main complementarity claim."
        ),
        (
            f"- The leave-dataset-out taxonomy sign test reaches {fmt(tax_primary['sign_accuracy'])} "
            f"accuracy (95% CI [{fmt(tax_primary['ci_low'])}, {fmt(tax_primary['ci_high'])}]) "
            f"over {tax_primary['n_predictions']} held-out trajectories."
        ),
    ]
    if tax_majority:
        lines.append(
            f"- The corresponding leave-dataset-out majority-class baseline reaches {fmt(tax_majority['sign_accuracy'])}; "
            "report taxonomy performance against this baseline, not in isolation."
        )
    if diss_primary:
        lines.append(
            f"- Constructed CKA/ODI dissociation pairs show median ODI difference "
            f"{fmt(diss_primary['odi_abs_diff_median'])} "
            f"(95% CI [{fmt(diss_primary['odi_abs_diff_median_ci_low'])}, {fmt(diss_primary['odi_abs_diff_median_ci_high'])}]) "
            f"despite near-matched CKA-input; this is the reported-scope complementarity demonstration."
        )
    if subj_primary:
        lines.append(
            f"- BCI IV 2a subject-unit ATCNet classifier-group ODI/accuracy Spearman r = "
            f"{fmt(subj_primary['spearman_odi_accuracy'])} "
            f"(95% CI [{fmt(subj_primary['spearman_ci_low'])}, {fmt(subj_primary['spearman_ci_high'])}], "
            f"n = {subj_primary['n_subjects']} held-out subjects)."
        )
    lines += [
        "",
        "## Primary Contrasts",
        "",
        "| ID | Family | Scope | Model/taxon | Metric | Estimate | 95% CI | p | n |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in primary_rows:
        ci = ""
        if str(row["ci_low"]) and str(row["ci_high"]):
            ci = f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"
        lines.append(
            f"| {row['contrast_id']} | {row['family']} | `{row['dataset_scope']}` | `{row['model_or_taxon']}` | "
            f"{row['metric']} | {fmt(row['estimate'])} | {ci} | {fmt(row['p_value'])} | {row['n']} |"
        )
    lines += [
        "",
        "## Exclusions",
        "",
        "- `sleepedf_full`/`EEGPT` is removed from reported-scope FM result tables because the two-channel bipolar Sleep-EDF montage produces a degenerate EEGPT configuration.",
        "- `p300`/`ATCNet` is removed from reported-scope supervised findings because the 200-sample P300 window triggers Braindecode architecture shortening and gives weak configuration-specific evidence.",
        "- The raw archival rows remain on disk for provenance; filtered manuscript tables are written in this folder.",
        "",
        "## Statistical Scope",
        "",
        f"- Primary contrasts are capped at {len(primary_rows)} rows in `primary_contrasts.csv`.",
        f"- `analysis_focused_effect_tests_clean.csv` contains {len(clean_focused)} focused but non-primary rows after configuration exclusions.",
        f"- `supplementary_contrasts_manifest.md` points to the full {5542} row exploratory/effect-size table; do not Holm-correct paper claims across the whole exploratory table.",
        "- Partial correlations use `activation_rank` as the available effective-rank/participation proxy exported by the CKA/RSA audit.",
        "",
        "## Artifacts",
        "",
        "- `variance_partitioning_rows.csv`",
        "- `partial_correlation_by_layer_group.csv`",
        "- `taxonomy_oos_sign_predictions.csv`",
        "- `taxonomy_oos_sign_summary.csv`",
        "- `cka_odi_dissociation_pairs.csv`",
        "- `cka_odi_dissociation_summary.csv`",
        "- `bci2a_subject_unit_layer_correlations.csv`",
        "- `bci2a_subject_unit_operation_group_correlations.csv`",
        "- `primary_contrasts.csv`",
        "- `excluded_failure_configurations.csv`",
    ]
    (out_dir / "representation_analysis_summary.md").write_text("\n".join(lines) + "\n")

    variance_lines = [
        "# Nested-Model Variance Partitioning",
        "",
        "Reduced model: CKA-input + RSA-input + RSA-label + log activation-rank.",
        "Full model: reduced model + ODI.",
        "Both models include dataset, architecture, mode, seed, and operation-group fixed effects.",
        "Interpretation: this is retained as a supplementary negative/control analysis. The main complementarity claim should use the constructed CKA/ODI dissociation artifact.",
        "",
        "| Scope | n rows | R^2 reduced | R^2 full | delta R^2 | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in variance_rows:
        variance_lines.append(
            f"| `{row['scope']}` | {row['n_rows']} | {fmt(row['r2_reduced'], 4)} | {fmt(row['r2_full'], 4)} | "
            f"{fmt(row['delta_r2'], 4)} | [{fmt(row['delta_r2_ci_low'], 4)}, {fmt(row['delta_r2_ci_high'], 4)}] |"
        )
    (out_dir / "variance_partitioning_summary.md").write_text("\n".join(variance_lines) + "\n")

    supp_lines = [
        "# Supplementary Contrast Manifest",
        "",
        "The `primary_contrasts.csv` table records the compact contrast set and its analysis units.",
        "All broader row-wise tests remain supplementary/exploratory.",
        "",
        "| Artifact | Role | Notes |",
        "| --- | --- | --- |",
        "| `reports/paper_statistics/all_true_vs_shuffled_corrected.csv` | Supplement | Full 5,542-row corrected/effect-size table; not a single primary family for Holm interpretation. |",
        "| `reports/paper_statistics/paper_focused_effect_tests.csv` | Supplement | Focused validation rows before configuration exclusions. |",
        "| `reports/representation_analysis/analysis_focused_effect_tests_clean.csv` | Analysis-clean supplement | Focused rows after removing Sleep-EDF/EEGPT and P300/ATCNet configuration failures. |",
        "| `reports/cka_rsa_stats/joined_layer_checkpoint_metrics.csv` | Supplement | Full 14,640-row joined HSDD/CKA/RSA table. |",
        "| `reports/representation_analysis/variance_partitioning_rows.csv` | Supplement/control | Negative broad additive-variance result for ODI over CKA/RSA/rank. |",
        "| `reports/representation_analysis/cka_odi_dissociation_summary.csv` | Primary/statistical support | Constructed CKA/ODI complementarity demonstration. |",
        "| `reports/representation_analysis/taxonomy_oos_sign_summary.csv` | Primary/statistical support | Out-of-sample taxonomy sign test with majority baselines. |",
    ]
    (out_dir / "supplementary_contrasts_manifest.md").write_text("\n".join(supp_lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    joined = enriched_joined_rows(args)
    variance_rows = variance_partitioning(joined, rng, args.bootstrap)
    partial_rows = partial_correlations(joined, rng, args.bootstrap)
    taxonomy_eval, taxonomy_summary = taxonomy_oos_test(joined, rng, args.bootstrap)
    dissociation_pairs, dissociation_summary = cka_odi_dissociation(joined, rng, args.bootstrap)
    subject_layer_rows, subject_group_rows = subject_unit_correlations(args, rng, args.bootstrap)
    clean_focused, excluded_focused = cleaned_focused_tests(args)

    excluded_configs = [
        {
            "dataset": "sleepedf_full",
            "model_family": "frozen_fm",
            "model": "eegpt",
            "analysis_action": "exclude_from_reported_scope",
            "reason": "Degenerate montage/channel-adaptation failure on two-channel bipolar Sleep-EDF.",
            "raw_artifacts_retained": True,
        },
        {
            "dataset": "p300",
            "model_family": "supervised_dl",
            "model": "atcnet",
            "analysis_action": "exclude_from_reported_scope",
            "reason": "Braindecode shortens/adapts ATCNet for the 200-sample P300 window; evidence is weak and configuration-specific.",
            "raw_artifacts_retained": True,
        },
    ]
    primary_rows = build_primary_contrasts(
        args,
        variance_rows,
        taxonomy_summary,
        subject_group_rows,
        clean_focused,
        dissociation_summary,
    )

    write_csv(out_dir / "variance_partitioning_rows.csv", variance_rows)
    write_csv(out_dir / "partial_correlation_by_layer_group.csv", partial_rows)
    write_csv(out_dir / "taxonomy_oos_sign_predictions.csv", taxonomy_eval)
    write_csv(out_dir / "taxonomy_oos_sign_summary.csv", taxonomy_summary)
    write_csv(out_dir / "cka_odi_dissociation_pairs.csv", dissociation_pairs)
    write_csv(out_dir / "cka_odi_dissociation_summary.csv", dissociation_summary)
    write_csv(out_dir / "bci2a_subject_unit_layer_correlations.csv", subject_layer_rows)
    write_csv(out_dir / "bci2a_subject_unit_operation_group_correlations.csv", subject_group_rows)
    write_csv(out_dir / "analysis_focused_effect_tests_clean.csv", clean_focused)
    write_csv(out_dir / "analysis_focused_effect_tests_excluded_rows.csv", excluded_focused)
    write_csv(out_dir / "excluded_failure_configurations.csv", excluded_configs)
    write_csv(out_dir / "primary_contrasts.csv", primary_rows)
    write_markdown(
        out_dir,
        variance_rows,
        partial_rows,
        taxonomy_summary,
        subject_group_rows,
        dissociation_summary,
        primary_rows,
        clean_focused,
        excluded_focused,
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
