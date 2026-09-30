"""Derived statistics for raw and zero-anchored band geometry.

Reads the compact records in ``results/``, including the representative-layer
measurements written by ``anchored/run_p300_label_control.py`` (P300) and
``anchored/run_anchored_heldout.py`` (BCI IV-2a, Sleep-EDF), writes the summary
statistics to ``results/anchored/statistics.json`` and derived tables to
``outputs/anchored/``.

    python anchored/analyses.py

Cluster bootstrap intervals resample trained networks (task x architecture x seed)
with replacement within each task, 5,000 draws, NumPy seed 2026.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RECORDS = RES / "anchored"
OUT = ROOT / "outputs" / "anchored"
DRAWS = 5000
RNG_SEED = 2026
KEY = ["dataset", "arch", "layer", "seed"]
ARCH_ORDER = ["eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet"]
TASKS = ["bci2a", "sleepedf_full", "seediv", "p300"]
LABELLED_TASKS = ["bci2a", "sleepedf_full", "p300"]
ATTENTION = ("eegconformer", "tsception", "atcnet")
FAR_ANCHOR_LAYERS = {("shallowconvnet", "pool_nonlin_exp"), ("shallowconvnet", "logits")}


def alpha_d(d):
    d = np.asarray(d, dtype=float)
    return np.exp(special.gammaln(d / 2) - special.gammaln((d + 1) / 2)) / np.sqrt(np.pi)


def read_measurements(path: Path, dataset: str, mode: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"label_mode": str}).rename(columns={
        "signed_baseline": "baseline", "signed_cross": "cross", "signed_residual": "residual"})
    frame["dataset"], frame["label_mode"] = dataset, mode
    return frame


def load_anchored() -> pd.DataFrame:
    seed_iv = pd.read_csv(RES / "seediv" / "fixed_hook_seed_layer_checkpoint_summary.csv").rename(columns={
        "baseline_sq_over_raw_den": "baseline", "cross_over_raw_den": "cross", "residual_dot_over_raw_den": "residual"})
    seed_iv["dataset"], seed_iv["label_mode"] = "seediv", "true"
    joined = pd.read_csv(RES / "taxonomy" / "joined_layer_checkpoint_metrics.csv")
    acc = joined[(joined["mode"] == "true") & (joined.dataset == "seediv")][["arch", "seed", "epoch", "val_acc"]].drop_duplicates()
    seed_iv = seed_iv.merge(acc, on=["arch", "seed", "epoch"], how="left")
    frames = [seed_iv]
    for mode in ("true", "shuffled"):
        for task in ("p300", "bci2a", "sleepedf_full"):
            frames.append(read_measurements(RECORDS / f"{task}_{mode}" / "layer_summary.csv", task, mode))
    cols = ["dataset", "label_mode", "arch", "layer", "seed", "epoch", "val_acc",
            "raw_odi", "anchored_odi", "baseline", "cross", "residual"]
    data = pd.concat([f[cols] for f in frames], ignore_index=True)

    full = joined[joined["mode"] == "true"][["dataset", "arch", "seed", "epoch", "layer", "hsdd_odi"]]
    data = data.merge(full.rename(columns={"hsdd_odi": "fullcache_raw_odi"}), on=["dataset", "arch", "seed", "epoch", "layer"], how="left")
    data.loc[data.label_mode == "shuffled", "fullcache_raw_odi"] = np.nan

    taxonomy = pd.read_csv(RES / "taxonomy" / "architecture_taxonomy_layer_rows.csv")
    groups = taxonomy[["arch", "layer", "operation_group"]].drop_duplicates()
    data = data.merge(groups, on=["arch", "layer"], how="left")
    dims = pd.read_csv(RES / "gaussian" / "analytic_gaussian_rows.csv")[["dataset", "arch", "layer", "output_dim"]]
    dims = dims.drop_duplicates().groupby(["dataset", "arch", "layer"]).output_dim.first().reset_index()
    data = data.merge(dims, on=["dataset", "arch", "layer"], how="left")
    data["alpha_d"] = alpha_d(data["output_dim"])
    data["final_epoch"] = data.groupby("dataset").epoch.transform("max")
    data["far_anchor"] = [(a, l) in FAR_ANCHOR_LAYERS for a, l in zip(data.arch, data.layer)]
    if data[["operation_group", "val_acc"]].isna().any().any():
        raise ValueError("unmapped operation group or missing accuracy")
    return data


def run_bootstrap(frame: pd.DataFrame, stat, cluster=("dataset", "arch", "seed")):
    """Percentile interval for ``stat(frame)``, resampling training runs within each task."""
    rng = np.random.default_rng(RNG_SEED)
    by_task = frame.groupby("dataset") if "dataset" in cluster and frame["dataset"].nunique() > 1 else [(None, frame)]
    runs = [[g for _, g in part.groupby(list(cluster))] for _, part in by_task]
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        parts = []
        for groups in runs:
            parts.extend(groups[j] for j in rng.integers(0, len(groups), len(groups)))
        draws[i] = stat(pd.concat(parts, ignore_index=True))
    return [float(stat(frame)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def task_mean(frame: pd.DataFrame, col: str) -> float:
    """Mean of ``col`` with each task weighted equally."""
    return float(frame.groupby("dataset")[col].mean().mean())


def endpoint_changes(data: pd.DataFrame) -> pd.DataFrame:
    k = ["label_mode"] + KEY
    start = data[data.epoch == 0].set_index(k)
    end = data[data.epoch == data.final_epoch].set_index(k)
    cols = ["raw_odi", "anchored_odi", "baseline"]
    delta = (end[cols] - start[cols]).dropna()
    delta["opposite"] = np.sign(delta.raw_odi) != np.sign(delta.anchored_odi)
    delta = delta.reset_index()
    delta["far_anchor"] = [(a, l) in FAR_ANCHOR_LAYERS for a, l in zip(delta.arch, delta.layer)]
    return delta


def coupling(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in data[data.label_mode == "true"].groupby(KEY + ["operation_group"]):
        g = g.sort_values("epoch")
        if len(g) < 3 or g.val_acc.std() < 1e-12:
            continue
        rec = dict(zip(KEY + ["operation_group"], key))
        for col, name in (("raw_odi", "r_raw"), ("anchored_odi", "r_anch")):
            rec[name] = np.corrcoef(g[col], g.val_acc)[0, 1] if g[col].std() > 1e-12 else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows).dropna()
    out["agree"] = np.sign(out.r_raw) == np.sign(out.r_anch)
    return out


def transfer(coup: pd.DataFrame, col: str, tasks: list[str]) -> pd.DataFrame:
    """Leave-one-task-out prediction of coupling signs by architecture and operation group."""
    rows = []
    coup = coup[coup.dataset.isin(tasks)]
    for held in tasks:
        train, test = coup[coup.dataset != held], coup[coup.dataset == held].copy()
        means = train.groupby(["arch", "operation_group"])[col].agg(["mean", "size"])
        means = means[(means["size"] >= 3) & (means["mean"].abs() > 1e-12)]
        signs = np.sign(means["mean"])
        majority = 1 if (train[col] > 0).mean() >= 0.5 else -1
        test["pred"] = [signs.get((a, g), np.nan) for a, g in zip(test.arch, test.operation_group)]
        test = test.dropna(subset=["pred"])
        test["tax_correct"] = (np.sign(test[col]) == test.pred).astype(float)
        test["maj_correct"] = (np.sign(test[col]) == majority).astype(float)
        test["majority_sign"] = majority
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def summarize_transfer(tr: pd.DataFrame) -> dict:
    return {
        "taxonomy": run_bootstrap(tr, lambda f: task_mean(f, "tax_correct")),
        "majority": run_bootstrap(tr, lambda f: task_mean(f, "maj_correct")),
        "difference": run_bootstrap(tr, lambda f: task_mean(f, "tax_correct") - task_mean(f, "maj_correct")),
        "n": int(len(tr)),
        "by_task": {d: {"taxonomy": float(g.tax_correct.mean()), "majority": float(g.maj_correct.mean()), "n": int(len(g)),
                        "majority_sign": int(g.majority_sign.iloc[0])} for d, g in tr.groupby("dataset")},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_anchored()
    data.to_csv(OUT / "anchored_analysis_table.csv", index=False)
    true = data[data.label_mode == "true"]
    s: dict[str, object] = {}

    s["scope"] = {f"{d}_{m}": {"rows": int(len(g)), "checkpoints": int(g.groupby(["arch", "seed", "epoch"]).ngroups),
                               "hooks": int(g.groupby(["arch", "layer"]).ngroups), "runs": int(g.groupby(["arch", "seed"]).ngroups)}
                  for (d, m), g in data.groupby(["dataset", "label_mode"])}
    checks = pd.concat([pd.read_csv(RECORDS / f"{t}_{m}" / "accuracy_checks.csv") for t in ("bci2a", "sleepedf_full", "p300") for m in ("true", "shuffled")])
    diff = (checks.recorded_val_acc - checks.recomputed_val_acc).abs()
    s["accuracy_recomputation"] = {"checkpoints": int(len(checks)), "exact": int((diff < 1e-9).sum()), "max_abs_diff": float(diff.max())}
    s["exclusions"] = {"raw_floor_pairs": 0, "anchor_excluded_pairs": 0}

    # Held-out raw ODI versus layer-survey raw ODI at identical checkpoints and layers.
    s["heldout_vs_fullcache"] = {
        d: {"n": int(len(g)), "pearson": float(stats.pearsonr(g.raw_odi, g.fullcache_raw_odi)[0]),
            "median_abs_diff": float((g.raw_odi - g.fullcache_raw_odi).abs().median())}
        for d, g in true.dropna(subset=["fullcache_raw_odi"]).groupby("dataset")}

    # Baseline term, raw/anchored association and the dimension reference.
    by_arch = true.groupby("arch").agg(
        baseline_median=("baseline", "median"), raw_mean=("raw_odi", "mean"), anchored_mean=("anchored_odi", "mean"),
        alpha_mean=("alpha_d", "mean"), n=("raw_odi", "size"))
    by_arch["anchored_excess"] = true.assign(x=true.anchored_odi - true.alpha_d).groupby("arch").x.mean()
    by_arch["anchored_above_alpha"] = true.assign(x=true.anchored_odi > true.alpha_d).groupby("arch").x.mean()
    s["baseline_median_all"] = float(true.baseline.median())
    s["baseline_median_by_family"] = {k: float(v) for k, v in true.groupby(true.arch.isin(ATTENTION).map({True: "attention_fusion", False: "compact_conv"})).baseline.median().items()}
    near = true[~true.far_anchor]
    s["baseline_median_by_family_excluding_far"] = {k: float(v) for k, v in near.groupby(near.arch.isin(ATTENTION).map({True: "attention_fusion", False: "compact_conv"})).baseline.median().items()}
    s["baseline_median_by_task_family"] = {f"{d}/{fam}": float(v) for (d, fam), v in near.groupby([near.dataset, near.arch.isin(ATTENTION).map({True: "attention_fusion", False: "compact_conv"})]).baseline.median().items()}
    s["far_anchor_fraction"] = float(true.far_anchor.mean())
    s["far_anchor_medians"] = {"baseline": float(true[true.far_anchor].baseline.median()),
                               "raw": float(true[true.far_anchor].raw_odi.median()),
                               "anchored": float(true[true.far_anchor].anchored_odi.median())}
    s["raw_anchored_spearman"] = {d: float(stats.spearmanr(g.raw_odi, g.anchored_odi)[0]) for d, g in true.groupby("dataset")}
    s["anchored_above_alpha_all"] = float((true.anchored_odi > true.alpha_d).mean())
    layer_means = true.groupby(["arch", "layer"])[["anchored_odi", "alpha_d", "baseline"]].mean()
    layer_means.to_csv(OUT / "anchored_layer_means.csv")
    s["layer_means"] = {f"{a}/{l}": {k: float(v) for k, v in r.items()} for (a, l), r in layer_means.iterrows()}
    by_task_arch = true.groupby(["dataset", "arch"]).agg(raw=("raw_odi", "mean"), anchored=("anchored_odi", "mean"),
                                                         baseline=("baseline", "median"), alpha=("alpha_d", "mean"))
    by_task_arch.to_csv(OUT / "anchored_by_task_architecture.csv")

    # Direction of change from initialization to the final checkpoint.
    delta = endpoint_changes(data)
    dt = delta[delta.label_mode == "true"]
    s["opposite_direction_true"] = run_bootstrap(dt, lambda f: f.opposite.mean())
    s["opposite_direction_n"] = int(len(dt))
    s["opposite_direction_by_task"] = {d: float(g.opposite.mean()) for d, g in dt.groupby("dataset")}
    per_arch = []
    for arch, g in dt.groupby("arch"):
        est, lo, hi = run_bootstrap(g, lambda f: f.opposite.mean())
        per_arch.append({"arch": arch, "n": len(g), "opposite": est, "ci_low": lo, "ci_high": hi,
                         "raw_change": g.raw_odi.mean(), "anchored_change": g.anchored_odi.mean(),
                         **{f"opposite_{d}": float(gg.opposite.mean()) for d, gg in g.groupby("dataset")}})
    pd.DataFrame(per_arch).to_csv(OUT / "opposite_direction_by_architecture.csv", index=False)
    task_arch = dt.groupby(["dataset", "arch"]).agg(opposite=("opposite", "mean"), n=("opposite", "size"),
                                                    raw_change=("raw_odi", "mean"), anchored_change=("anchored_odi", "mean"))
    task_arch.to_csv(OUT / "opposite_direction_by_task_architecture.csv")
    s["opposite_by_task_arch"] = {f"{d}/{a}": {k: float(v) for k, v in r.items()} for (d, a), r in task_arch.iterrows()}
    # Sensitivity: require both endpoint changes to exceed a minimum absolute size.
    threshold_rows = []
    for thr in (0.0, 0.001, 0.005, 0.01):
        keep = dt[(dt.raw_odi.abs() > thr) & (dt.anchored_odi.abs() > thr)]
        est, lo, hi = run_bootstrap(keep, lambda f: f.opposite.mean())
        threshold_rows.append({"threshold": thr, "n": int(len(keep)), "opposite": est, "ci_low": lo, "ci_high": hi,
                               **{f"opposite_{a}": float(g.opposite.mean()) for a, g in keep.groupby("arch")}})
    pd.DataFrame(threshold_rows).to_csv(OUT / "opposite_direction_thresholds.csv", index=False)
    s["opposite_direction_thresholds"] = threshold_rows
    hooks = dt.groupby(["dataset", "arch", "layer"])[["raw_odi", "anchored_odi"]].mean()
    hooks["opposite"] = np.sign(hooks.raw_odi) != np.sign(hooks.anchored_odi)
    hooks.to_csv(OUT / "hook_mean_changes.csv")
    s["hooks_opposite"] = [int(hooks.opposite.sum()), int(len(hooks))]

    # Accuracy coupling and cross-task transfer.
    coup = coupling(data)
    coup.to_csv(OUT / "accuracy_coupling_trajectories.csv", index=False)
    s["coupling_sign_agreement"] = run_bootstrap(coup, lambda f: f.agree.mean())
    s["coupling_n"] = int(len(coup))
    s["coupling_agreement_by_arch"] = {a: float(g.agree.mean()) for a, g in coup.groupby("arch")}
    s["coupling_agreement_by_task_arch"] = {f"{d}/{a}": float(g.agree.mean()) for (d, a), g in coup.groupby(["dataset", "arch"])}
    s["coupling_mean_r_by_task_arch"] = {f"{d}/{a}": {"raw": float(g.r_raw.mean()), "anchored": float(g.r_anch.mean())}
                                         for (d, a), g in coup.groupby(["dataset", "arch"])}
    for col, name in (("r_raw", "raw"), ("r_anch", "anchored")):
        for tasks, label in ((TASKS, "four"), ([t for t in TASKS if t != "sleepedf_full"], "three")):
            tr = transfer(coup, col, tasks)
            tr.to_csv(OUT / f"cross_task_transfer_{name}_{label}.csv", index=False)
            s[f"transfer_{name}_{label}"] = summarize_transfer(tr)

    # Label control: true versus shuffled labels with shared initialization.
    labelled = data[data.dataset.isin(LABELLED_TASKS)]
    hook_mean = labelled.groupby(["dataset", "label_mode", "arch", "seed", "epoch"])[["raw_odi", "anchored_odi", "val_acc"]].mean().reset_index()
    hook_mean.to_csv(OUT / "label_control_trajectories.csv", index=False)
    final = hook_mean[hook_mean.epoch == hook_mean.groupby("dataset").epoch.transform("max")]
    control = []
    for (task, arch), g in final.groupby(["dataset", "arch"]):
        wide = g.pivot(index="seed", columns="label_mode", values=["raw_odi", "anchored_odi", "val_acc"]).dropna()
        for metric in ("raw_odi", "anchored_odi", "val_acc"):
            d = wide[(metric, "true")] - wide[(metric, "shuffled")]
            control.append({"dataset": task, "arch": arch, "metric": metric, "n_seeds": len(d),
                            "true": wide[(metric, "true")].mean(), "shuffled": wide[(metric, "shuffled")].mean(),
                            "true_minus_shuffled": d.mean(), "seeds_positive": int((d > 0).sum())})
    control = pd.DataFrame(control)
    control.to_csv(OUT / "label_control_final.csv", index=False)
    anch = control[control.metric == "anchored_odi"]
    accd = control[control.metric == "val_acc"]
    s["label_control_anchored_abs_max"] = float(anch.true_minus_shuffled.abs().max())
    s["label_control_anchored_abs_median"] = float(anch.true_minus_shuffled.abs().median())
    s["label_control_accuracy_range"] = [float(accd.true_minus_shuffled.min()), float(accd.true_minus_shuffled.max())]
    s["label_control_consistent"] = [
        {"dataset": r.dataset, "arch": r.arch, "diff": float(r.true_minus_shuffled), "positive": int(r.seeds_positive), "n": int(r.n_seeds)}
        for r in anch.itertuples() if r.seeds_positive in (0, r.n_seeds)]
    # Seed-level differences, and the compact architectures without the far-anchor layers.
    seed_level = {}
    for label, frame in (("all_layers", labelled), ("excluding_far_anchor", labelled[~labelled.far_anchor])):
        per_seed = frame.groupby(["dataset", "label_mode", "arch", "seed", "epoch"])[["anchored_odi", "val_acc"]].mean().reset_index()
        per_seed = per_seed[per_seed.epoch == per_seed.groupby("dataset").epoch.transform("max")]
        wide = per_seed.pivot_table(index=["dataset", "arch", "seed"], columns="label_mode", values=["anchored_odi", "val_acc"]).dropna()
        d_anch = wide[("anchored_odi", "true")] - wide[("anchored_odi", "shuffled")]
        d_acc = wide[("val_acc", "true")] - wide[("val_acc", "shuffled")]
        entry = {"pairs": int(len(d_anch)), "anchored_abs_median": float(d_anch.abs().median()),
                 "anchored_abs_max": float(d_anch.abs().max()), "accuracy_range": [float(d_acc.min()), float(d_acc.max())]}
        for arch in ("eegnet", "shallowconvnet"):
            da = d_anch.xs(arch, level="arch")
            entry[arch] = {"pairs": int(len(da)), "positive": int((da > 0).sum()), "negative": int((da < 0).sum()),
                           "task_means": {d: float(v) for d, v in da.groupby(level="dataset").mean().items()}}
        seed_level[label] = entry
    s["label_control_seed_level"] = seed_level
    dl = delta[delta.dataset.isin(LABELLED_TASKS)]
    s["opposite_by_label_mode"] = {m: run_bootstrap(g, lambda f: f.opposite.mean()) for m, g in dl.groupby("label_mode")}
    s["opposite_by_label_mode_task"] = {f"{m}/{d}": float(g.opposite.mean()) for (m, d), g in dl.groupby(["label_mode", "dataset"])}
    s["opposite_eegconformer_by_label_mode"] = {m: float(g[g.arch == "eegconformer"].opposite.mean()) for m, g in dl.groupby("label_mode")}
    s["baseline_median_by_label_mode"] = {m: float(g.baseline.median()) for m, g in labelled.groupby("label_mode")}
    # Layerwise anchored change, true versus shuffled, in the held-out analysis.
    for col, name in (("anchored_odi", "anchored"), ("raw_odi", "raw")):
        lw = dl.groupby(["dataset", "arch", "layer", "label_mode"])[col].mean().unstack("label_mode").dropna()
        lw.to_csv(OUT / f"{name}_change_true_vs_shuffled.csv")
        s[f"{name}_change_true_vs_shuffled"] = {
            "layers": int(len(lw)), "pearson": float(stats.pearsonr(lw["true"], lw["shuffled"])[0]),
            "sign_agreement": float((np.sign(lw["true"]) == np.sign(lw["shuffled"])).mean()),
            "mean_abs_true": float(lw["true"].abs().mean()), "mean_abs_diff": float((lw["true"] - lw["shuffled"]).abs().mean()),
            "ratio": float((lw["true"] - lw["shuffled"]).abs().mean() / lw["true"].abs().mean())}
    s["coupling_positive_fraction"] = {"raw": float((coup.r_raw > 0).mean()), "anchored": float((coup.r_anch > 0).mean()),
                                       "raw_by_task": {d: float((g.r_raw > 0).mean()) for d, g in coup.groupby("dataset")},
                                       "anchored_by_task": {d: float((g.r_anch > 0).mean()) for d, g in coup.groupby("dataset")}}
    s["label_control_by_task"] = {
        d: {"anchored_abs_mean": float(g[g.metric == "anchored_odi"].true_minus_shuffled.abs().mean()),
            "raw_abs_mean": float(g[g.metric == "raw_odi"].true_minus_shuffled.abs().mean()),
            "acc_min": float(g[g.metric == "val_acc"].true_minus_shuffled.min()),
            "acc_max": float(g[g.metric == "val_acc"].true_minus_shuffled.max())} for d, g in control.groupby("dataset")}
    patch = labelled[(labelled.arch == "eegconformer") & (labelled.layer == "patch_embedding")]
    s["conformer_patch"] = {
        f"{d}/{m}/epoch{e}": {"raw": float(g.raw_odi.mean()), "anchored": float(g.anchored_odi.mean())}
        for (d, m, e), g in patch.groupby(["dataset", "label_mode", "epoch"]) if e in (0, g.final_epoch.iloc[0])}
    pw = patch.pivot_table(index=["dataset", "label_mode", "seed"], columns="epoch", values=["raw_odi", "anchored_odi"])
    directions = {}
    for (d, m), g in pw.groupby(level=[0, 1]):
        g = g.dropna(axis=1, how="all")
        last = max(c[1] for c in g.columns)
        directions[f"{d}/{m}"] = {"raw_decrease": int(((g[("raw_odi", last)] - g[("raw_odi", 0)]) < 0).sum()),
                                  "anchored_increase": int(((g[("anchored_odi", last)] - g[("anchored_odi", 0)]) > 0).sum()),
                                  "n": int(len(g))}
    s["conformer_patch_seed_directions"] = directions

    # Four-task layer survey: raw ODI change, true versus shuffled labels.
    joined = pd.read_csv(RES / "taxonomy" / "joined_layer_checkpoint_metrics.csv")
    joined = joined[~((joined.dataset == "p300") & (joined.arch == "atcnet"))]
    last = joined.groupby("dataset").epoch.transform("max")
    ends = joined[(joined.epoch == 0) | (joined.epoch == last)].copy()
    ends["point"] = np.where(ends.epoch == 0, "start", "end")
    wide = ends.pivot_table(index=["dataset", "arch", "layer", "mode", "seed"], columns="point", values="hsdd_odi")
    wide["change"] = wide["end"] - wide["start"]
    hook_change = wide.groupby(["dataset", "arch", "layer", "mode"]).change.mean().unstack("mode").dropna()
    hook_change.to_csv(OUT / "four_task_raw_change_true_vs_shuffled.csv")
    s["four_task_raw_change_true_vs_shuffled"] = {
        "hooks": int(len(hook_change)), "pearson": float(stats.pearsonr(hook_change["true"], hook_change["shuffled"])[0]),
        "sign_agreement": float((np.sign(hook_change["true"]) == np.sign(hook_change["shuffled"])).mean()),
        "mean_abs_true_change": float(hook_change["true"].abs().mean()),
        "mean_abs_true_minus_shuffled": float((hook_change["true"] - hook_change["shuffled"]).abs().mean()),
        "by_dataset_pearson": {d: float(stats.pearsonr(g["true"], g["shuffled"])[0]) for d, g in hook_change.groupby(level="dataset")}}

    # Trajectory summaries for figures.
    true.groupby(["dataset", "arch", "seed", "epoch"])[["raw_odi", "anchored_odi", "baseline"]].mean().reset_index() \
        .to_csv(OUT / "architecture_trajectories_by_seed.csv", index=False)

    # Supervised accuracy: best checkpoint and final checkpoint (no selection).
    tax = pd.read_csv(RES / "taxonomy" / "architecture_taxonomy_layer_rows.csv")
    acc = tax[["dataset", "arch", "mode", "seed", "best_val_acc", "final_val_acc", "final_epoch"]].drop_duplicates()
    acc.groupby(["dataset", "arch", "mode"]).agg(
        best_mean=("best_val_acc", "mean"), best_sd=("best_val_acc", "std"),
        final_mean=("final_val_acc", "mean"), final_sd=("final_val_acc", "std"),
        final_epoch=("final_epoch", "first"), n_seeds=("seed", "nunique")).reset_index() \
        .to_csv(OUT / "accuracy_best_and_final.csv", index=False)

    by_arch["opposite"] = pd.DataFrame(per_arch).set_index("arch").opposite
    by_arch["coupling_agreement"] = pd.Series(s["coupling_agreement_by_arch"])
    by_arch.reindex(ARCH_ORDER).to_csv(OUT / "baseline_by_architecture.csv")
    (RECORDS / "statistics.json").write_text(json.dumps(s, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: s[k] for k in ("scope", "accuracy_recomputation", "baseline_median_by_family_excluding_far",
                                         "opposite_direction_true", "coupling_sign_agreement")}, indent=1, default=float))


if __name__ == "__main__":
    main()
