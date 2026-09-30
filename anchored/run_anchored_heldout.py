"""Raw and zero-anchored band coherence on fixed held-out windows for BCI IV-2a and Sleep-EDF.

Uses the same design as for SEED-IV and P300: a fixed class-balanced set of
held-out windows, fixed representative layers, every saved training epoch of every
seed, and both label conditions. The held-out accuracy of each saved model is
recomputed from its weights and stored next to the value recorded during training.

    python anchored/run_anchored_heldout.py --dataset bci2a --project /path/to/project \
        --bci-root /path/to/bci_iv_2a_fif --label-mode true --out results/anchored/bci2a_true
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipelines")]


def _alias_moabb_names() -> None:
    """Map pre-1.1 MOABB dataset names (e.g. BNCI2014001) onto their current names.

    Braindecode 1.2 imports its MOABB wrappers at package import time; only the
    model classes are used here.
    """
    try:
        import moabb.datasets as datasets
    except ImportError:
        return

    def lookup(name: str):
        alternative = re.sub(r"^([A-Za-z]+)(\d{4})(\d{3})$", r"\1\2_\3", name)
        if alternative != name and hasattr(datasets, alternative):
            return getattr(datasets, alternative)
        raise AttributeError(name)

    datasets.__getattr__ = lookup


_alias_moabb_names()

from hsdd.synthetic import fft_band_components  # noqa: E402
from hsrg.audit_metrics import paired_band_metrics  # noqa: E402
from train_eegnet_bci2a_layer_audit import LayerCapture  # noqa: E402

FIXED_HOOKS = {
    "eegnet": ("conv_temporal", "bnorm_1", "conv_spatial", "conv_separable_depth", "bnorm_2", "pool_2", "logits"),
    "shallowconvnet": ("conv_time_spat", "bnorm", "conv_nonlin_exp", "pool", "pool_nonlin_exp", "logits"),
    "eegconformer": ("patch_embedding", "transformer.0", "transformer.2", "transformer.5", "fc", "logits"),
    "tsception": ("temporal_blocks.0", "temporal_blocks.2", "spatial_block_1", "spatial_block_2",
                  "batch_temporal_lay", "batch_spatial_lay", "dense_layer", "logits"),
    "atcnet": ("conv_block", "attention_blocks.0", "attention_blocks.4", "temporal_conv_nets.0",
               "temporal_conv_nets.4", "final_layer.4", "logits"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bci2a", "sleepedf_full"), required=True)
    parser.add_argument("--project", type=Path, required=True, help="Directory holding reports/ and prepared/.")
    parser.add_argument("--bci-root", type=Path, default=None,
                        help="BCI IV-2a FIF directory, used when the saved training arguments do not name one.")
    parser.add_argument("--label-mode", choices=("true", "shuffled"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archs", default="eegnet,shallowconvnet,eegconformer,tsception,atcnet")
    parser.add_argument("--seeds", default="41,42,43,44,45,46")
    parser.add_argument("--per-class", type=int, default=None, help="Held-out windows per class (default 16 for BCI, 13 for Sleep-EDF).")
    parser.add_argument("--batch-windows", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def run_paths(project: Path, dataset: str, arch: str, mode: str, seed: int) -> tuple[Path, Path]:
    reports = project / "reports"
    if dataset == "bci2a":
        suffix = "" if seed == 41 else f"_seed{seed}"
        return reports / f"{arch}_bci2a_dynamics_{mode}{suffix}_checkpoints", reports / f"{arch}_bci2a_dynamics_{mode}{suffix}.json"
    return (reports / f"sleepedf_full_{arch}_dynamics_{mode}_seed{seed}_checkpoints",
            reports / f"sleepedf_full_{arch}_dynamics_{mode}_seed{seed}.json")


def recorded_accuracy(report: Path) -> dict[int, float]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {int(item["epoch"]): float(item["val_acc"]) for item in payload["audits"]}


def load_heldout(dataset: str, project: Path, ckpt_args: dict, mode: str, seed: int, bci_root: Path | None = None):
    """Return held-out arrays exactly as constructed by the training script."""
    if dataset == "bci2a":
        from train_eegnet_bci2a_layer_audit import load_bci2a, make_train_val

        root = ckpt_args.get("root") or bci_root
        if root is None:
            raise ValueError("pass --bci-root: the saved training arguments do not name the BCI IV-2a directory")
        data = load_bci2a(Path(root), tmin=ckpt_args.get("tmin", 0.0), tmax=ckpt_args.get("tmax", 3.0))
        _, _, val_x, val_y = make_train_val(data, label_mode=mode, seed=seed)
        return val_x.astype(np.float32), val_y.astype(np.int64), 1.0 / data["dt"]
    from train_sleepedf_arch_layer_audit import load_cache

    cache = Path(ckpt_args["cache"])
    cache = cache if cache.is_absolute() else project / cache
    data = load_cache(cache, ckpt_args["val_recordings"], seed, ckpt_args["max_train_per_class"], ckpt_args["max_val_per_class"])
    return data["val_x"].astype(np.float32), data["val_y"].astype(np.int64), float(data["sfreq"])


def build_model(dataset: str, arch: str, n_chans: int, n_times: int, n_outputs: int, sfreq: float):
    if dataset == "bci2a":
        from train_bci2a_arch_layer_audit import arch_spec

        return arch_spec(arch, n_chans, n_times, n_outputs=n_outputs, sfreq=sfreq)
    from train_sleepedf_arch_layer_audit import arch_spec

    return arch_spec(arch, n_chans, n_times, n_outputs=n_outputs)


def accuracy(model, x: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 64) -> float:
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(torch.from_numpy(x[start:start + batch]).to(device))
            correct += int((logits.argmax(1).cpu().numpy() == y[start:start + batch]).sum())
    return correct / len(x)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    archs = [a for a in args.archs.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    per_class = args.per_class or (16 if args.dataset == "bci2a" else 13)

    heldout_cache: dict[int, tuple] = {}
    picked = components = None
    summary, checks, missing = [], [], []
    started = time.time()
    for arch in archs:
        for seed in seeds:
            ckpt_dir, report = run_paths(args.project, args.dataset, arch, args.label_mode, seed)
            files = sorted(ckpt_dir.glob(f"{arch}_{args.label_mode}_epoch_*.pt")) if ckpt_dir.exists() else []
            if not files or not report.exists():
                missing.append({"arch": arch, "seed": seed, "reason": "no checkpoint directory or report"})
                continue
            recorded = recorded_accuracy(report)
            for ckpt_path in files:
                epoch = int(ckpt_path.stem.rsplit("_", 1)[-1])
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_args = ckpt.get("args", {})
                if int(ckpt_args.get("seed", seed)) != seed:
                    raise ValueError(f"{ckpt_path} was trained with seed {ckpt_args.get('seed')}, expected {seed}")
                if seed not in heldout_cache:
                    heldout_cache[seed] = load_heldout(args.dataset, args.project, ckpt_args, args.label_mode, seed, args.bci_root)
                val_x, val_y, sfreq = heldout_cache[seed]
                if picked is None:
                    ref_x, ref_y, _ = heldout_cache[seed] if seed == 41 else load_heldout(
                        args.dataset, args.project, ckpt_args, args.label_mode, 41, args.bci_root)
                    rng = np.random.default_rng(41)
                    idx = []
                    for label in sorted(set(ref_y.tolist())):
                        ids = np.flatnonzero(ref_y == label)
                        idx.extend(rng.choice(ids, per_class, replace=False).tolist())
                    picked = np.asarray(sorted(idx))
                    windows = ref_x[picked]
                    components = np.stack([fft_band_components(w, sfreq=sfreq)[0] for w in windows]).astype(np.float32)
                    (args.out / "selection.json").write_text(json.dumps({
                        "dataset": args.dataset, "per_class": per_class, "n_windows": int(len(picked)),
                        "heldout_indices": picked.tolist(), "labels": ref_y[picked].tolist(), "sfreq": sfreq,
                        "rng": "numpy.default_rng(41), per class choice without replacement, sorted"}, indent=2), encoding="utf-8")
                n_outputs = int(len(np.unique(val_y)))
                model, layers, _ = build_model(args.dataset, arch, val_x.shape[1], val_x.shape[2], n_outputs, sfreq)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                model.to(device).eval()
                recomputed = accuracy(model, val_x, val_y, device)
                names = tuple(n for n in FIXED_HOOKS[arch] if n in layers or n == "logits")
                capture = LayerCapture(model, names)
                per_layer = {n: [] for n in names}
                try:
                    for start in range(0, len(components), args.batch_windows):
                        batch = components[start:start + args.batch_windows]
                        with torch.no_grad():
                            captured = capture(batch.reshape((-1,) + batch.shape[2:]), device=device)
                            zero = capture(np.zeros((1,) + batch.shape[2:], np.float32), device=device)
                        for n in names:
                            feats = np.asarray(captured[n]).reshape(len(batch), 5, -1).astype(np.float64)
                            base = np.asarray(zero[n])[0].reshape(-1).astype(np.float64)
                            per_layer[n].extend(paired_band_metrics(feats, base, eps=args.eps))
                finally:
                    capture.close()
                for n in names:
                    s = per_layer[n]
                    summary.append({
                        "dataset": args.dataset, "label_mode": args.label_mode, "arch": arch, "layer": n,
                        "seed": seed, "epoch": epoch, "n_windows": len(s), "val_acc": recorded.get(epoch, float("nan")),
                        "raw_odi": float(np.mean([r["raw_odi"] for r in s])),
                        "anchored_odi": float(np.nanmean([r["anchored_odi"] for r in s])),
                        "signed_baseline": float(np.mean([r["baseline_sq_over_raw_den"] for r in s])),
                        "signed_cross": float(np.mean([r["cross_over_raw_den"] for r in s])),
                        "signed_residual": float(np.mean([r["residual_over_raw_den"] for r in s])),
                        "raw_floor_pairs": int(sum(r["raw_floor_pair_count"] for r in s)),
                        "anchor_excluded_pairs": int(sum(r["anchor_excluded_pair_count"] for r in s)),
                    })
                checks.append({"dataset": args.dataset, "label_mode": args.label_mode, "arch": arch, "seed": seed,
                               "epoch": epoch, "recorded_val_acc": recorded.get(epoch, float("nan")),
                               "recomputed_val_acc": recomputed})
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(args.dataset, args.label_mode, arch, seed, epoch, f"acc {recomputed:.4f}/{recorded.get(epoch, float('nan')):.4f}",
                      f"{time.time() - started:.0f}s", flush=True)

    for name, rows in (("layer_summary.csv", summary), ("accuracy_checks.csv", checks), ("missing_runs.csv", missing)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0]) if rows else ["arch", "seed", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "runtime.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "tf32": False, "seconds": time.time() - started}, indent=2), encoding="utf-8")
    print(f"done: {len(checks)} checkpoints, {len(missing)} missing runs, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
