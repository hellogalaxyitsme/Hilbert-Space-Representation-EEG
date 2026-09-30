"""Raw and zero-anchored band coherence for true- and shuffled-label P300 models.

Evaluates the same 64 fixed class-balanced held-out windows (subjects 9 and 10)
and the same representative layers used by ``pipelines/run_p300_paired_audit.py``,
for a chosen label mode. The validation accuracy of each saved model is
recomputed and checked against the accuracy recorded during training before any
coherence value is written.

Example::

    python anchored/run_p300_label_control.py --cache path/to/p300_reconstructed.npz \
        --reports path/to/reports --label-mode shuffled --out results/anchored/p300_shuffled
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipelines")]
from p300_utils import load_p300_cache  # noqa: E402
from train_bci2a_arch_layer_audit import arch_spec  # noqa: E402
from train_eegnet_bci2a_layer_audit import LayerCapture  # noqa: E402
from hsdd.synthetic import fft_band_components  # noqa: E402
from hsrg.audit_metrics import paired_band_metrics  # noqa: E402
from run_p300_paired_audit import (  # noqa: E402
    FIXED_HOOKS,
    fixed_balanced,
    model_accuracy,
    recorded_val_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-mode", choices=("true", "shuffled"), required=True)
    parser.add_argument("--arch", choices=tuple(FIXED_HOOKS))
    parser.add_argument("--seeds", default="41,42,43,44,45,46")
    parser.add_argument("--epochs", default="0,1,2,5,10,20")
    parser.add_argument("--batch-windows", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    architectures = (args.arch,) if args.arch else tuple(FIXED_HOOKS)
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    epochs = [int(v) for v in args.epochs.split(",") if v.strip()]
    mode = args.label_mode

    data = load_p300_cache(args.cache, val_subjects="9,10", seed=41, max_train_per_class=288, max_val_per_class=72)
    pick = fixed_balanced(data, 41)
    x = data["val_x"][pick]
    components = np.stack([fft_band_components(v, sfreq=data["sfreq"])[0] for v in x]).astype(np.float32)

    summary_rows, check_rows, missing = [], [], []
    started = time.time()
    for arch in architectures:
        names = FIXED_HOOKS[arch]
        for seed in seeds:
            run_dir = args.reports / f"p300_{arch}_dynamics_{mode}_seed{seed}_checkpoints"
            report = args.reports / f"p300_{arch}_dynamics_{mode}_seed{seed}.json"
            run_data = load_p300_cache(args.cache, val_subjects="9,10", seed=seed, max_train_per_class=288, max_val_per_class=72)
            for epoch in epochs:
                checkpoint = run_dir / f"{arch}_{mode}_epoch_{epoch:03d}.pt"
                if not checkpoint.exists():
                    missing.append((arch, seed, epoch))
                    continue
                model, layers, _ = arch_spec(arch, x.shape[1], x.shape[2], n_outputs=2, sfreq=data["sfreq"])
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
                model.eval()
                recomputed = model_accuracy(model, run_data["val_x"], run_data["val_y"])
                recorded = recorded_val_accuracy(report, epoch)
                if abs(recomputed - recorded) > 1e-6:
                    raise AssertionError((arch, seed, epoch, recomputed, recorded))
                capture = LayerCapture(model, names)
                per_layer = {name: [] for name in names}
                try:
                    for offset in range(0, len(x), args.batch_windows):
                        stop = min(offset + args.batch_windows, len(x))
                        batch = components[offset:stop]
                        captured = capture(batch.reshape((-1,) + batch.shape[2:]), device=torch.device("cpu"))
                        zero = capture(np.zeros((1,) + batch.shape[2:], np.float32), device=torch.device("cpu"))
                        for name in names:
                            feats = captured[name].reshape(stop - offset, 5, -1).astype(np.float64)
                            base = zero[name][0].reshape(-1).astype(np.float64)
                            per_layer[name].extend(paired_band_metrics(feats, base, eps=args.eps))
                finally:
                    capture.close()
                    del model
                    gc.collect()
                for name in names:
                    scores = per_layer[name]
                    summary_rows.append({
                        "label_mode": mode, "arch": arch, "layer": name, "seed": seed, "epoch": epoch,
                        "n_windows": len(scores), "val_acc": recorded,
                        "raw_odi": float(np.mean([s["raw_odi"] for s in scores])),
                        "anchored_odi": float(np.nanmean([s["anchored_odi"] for s in scores])),
                        "signed_baseline": float(np.mean([s["baseline_sq_over_raw_den"] for s in scores])),
                        "signed_cross": float(np.mean([s["cross_over_raw_den"] for s in scores])),
                        "signed_residual": float(np.mean([s["residual_over_raw_den"] for s in scores])),
                        "raw_floor_pairs": int(sum(s["raw_floor_pair_count"] for s in scores)),
                        "anchor_excluded_pairs": int(sum(s["anchor_excluded_pair_count"] for s in scores)),
                    })
                check_rows.append({"label_mode": mode, "arch": arch, "seed": seed, "epoch": epoch,
                                   "recorded_val_acc": recorded, "recomputed_val_acc": recomputed})
                print(mode, arch, seed, epoch, round(time.time() - started, 1), flush=True)

    for filename, rows in (("layer_summary.csv", summary_rows), ("accuracy_checks.csv", check_rows)):
        with (args.out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.out / "missing_checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arch", "seed", "epoch"])
        writer.writerows(missing)
    print(f"{len(check_rows)} checkpoints, {len(missing)} missing, {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
