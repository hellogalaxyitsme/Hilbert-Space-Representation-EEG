#!/usr/bin/env python3
"""Recompute fixed-hook P300 raw and anchored checkpoint measures.

The command needs a locally prepared public BNCI 2014-009 cache and
compatible locally held checkpoints. Neither is distributed in this archive.
It uses the same 64 fixed class-balanced held-out windows for every available
architecture, seed, and checkpoint, and retains every declared fixed hook.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipelines")]
from p300_utils import load_p300_cache
from train_bci2a_arch_layer_audit import arch_spec
from train_eegnet_bci2a_layer_audit import LayerCapture
from hsdd.synthetic import fft_band_components
from hsrg.audit_metrics import paired_band_metrics
from hsrg.checkpoint_shards import fingerprint, load_verified_shard, write_verified_shard


FIXED_HOOKS = {
    "eegnet": ("conv_temporal", "bnorm_1", "conv_spatial", "conv_separable_depth", "bnorm_2", "pool_2", "logits"),
    "shallowconvnet": ("conv_time_spat", "bnorm", "conv_nonlin_exp", "pool", "pool_nonlin_exp", "logits"),
    "eegconformer": ("patch_embedding", "transformer.0", "transformer.2", "transformer.5", "fc", "logits"),
    "tsception": ("temporal_blocks.0", "temporal_blocks.2", "spatial_block_1", "spatial_block_2", "batch_temporal_lay", "batch_spatial_lay", "dense_layer", "logits"),
}


def _close_or_both_nan(left, right, *, rtol=1e-6, atol=1e-6):
    return bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_fingerprint(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "schema": "p300-paired-audit-shard-v2",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if cuda_available else None,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32 if cuda_available else None,
    }


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True, help="Directory containing locally held reports and checkpoints.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=("eegnet", "shallowconvnet", "eegconformer", "tsception"))
    parser.add_argument("--seeds", default="41,42,43,44,45,46")
    parser.add_argument("--epochs", default="0,1,2,5,10,20")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-windows", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--skip-missing", action="store_true", help="Record and skip missing requested checkpoint files instead of failing before inference.")
    return parser.parse_args()


def fixed_balanced(data: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for label in sorted(set(data["val_y"].tolist())):
        ids = np.flatnonzero(data["val_y"] == label)
        if len(ids) < 32:
            raise ValueError(f"validation class {label} has {len(ids)}, requires 32")
        picks.extend(rng.choice(ids, 32, replace=False).tolist())
    return np.asarray(sorted(picks))


def requested_combinations(architectures, seeds, epochs):
    return [(arch, seed, epoch) for arch in architectures for seed in seeds for epoch in epochs]


def checkpoint_path(reports: Path, arch: str, seed: int, epoch: int) -> Path:
    return reports / f"p300_{arch}_dynamics_true_seed{seed}_checkpoints" / f"{arch}_true_epoch_{epoch:03d}.pt"


def preflight_checkpoints(reports: Path, architectures, seeds, epochs):
    available = []
    missing = []
    for arch, seed, epoch in requested_combinations(architectures, seeds, epochs):
        path = checkpoint_path(reports, arch, seed, epoch)
        record = {"arch": arch, "seed": seed, "epoch": epoch, "checkpoint": str(path)}
        (available if path.exists() else missing).append(record)
    return available, missing


def shard_path(out: Path, arch: str, seed: int, epoch: int) -> Path:
    return out / "checkpoint_shards" / f"{arch}_seed{seed}_epoch{epoch:03d}.json"


def recorded_val_accuracy(report: Path, epoch: int) -> float:
    for item in json.loads(report.read_text(encoding="utf-8"))["audits"]:
        if int(item["epoch"]) == epoch:
            return float(item["val_acc"])
    raise KeyError(epoch)


def model_accuracy(model, x: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).detach().cpu().numpy()
    return float((logits.argmax(1) == y).mean())


def main() -> None:
    audit_args = args()
    audit_args.out.mkdir(parents=True, exist_ok=True)
    architectures = (audit_args.arch,) if audit_args.arch else tuple(FIXED_HOOKS)
    seeds = tuple(int(value) for value in audit_args.seeds.split(",") if value.strip())
    epochs = tuple(int(value) for value in audit_args.epochs.split(",") if value.strip())
    if not seeds or not epochs or len(set(seeds)) != len(seeds) or len(set(epochs)) != len(epochs):
        raise ValueError("seeds and epochs must be nonempty and unique")
    available, missing = preflight_checkpoints(audit_args.reports, architectures, seeds, epochs)
    coverage = {"requested": len(available) + len(missing), "available": len(available), "missing": missing, "skip_missing": audit_args.skip_missing}
    (audit_args.out / "p300_checkpoint_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    if missing and not audit_args.skip_missing:
        raise FileNotFoundError(f"{len(missing)} requested checkpoints are missing; rerun with --skip-missing to preserve available scope")
    available_keys = {(item["arch"], item["seed"], item["epoch"]) for item in available}
    data = load_p300_cache(audit_args.cache, val_subjects="9,10", seed=41, max_train_per_class=288, max_val_per_class=72)
    cache_digest = sha256(audit_args.cache)
    pick = fixed_balanced(data, audit_args.seed)
    x = data["val_x"][pick]
    y = data["val_y"][pick]
    components = np.stack([fft_band_components(value, sfreq=data["sfreq"])[0] for value in x]).astype(np.float32)
    metadata = {"cache_sha256": cache_digest, "fixed_balanced_cache_indices": pick.tolist(), "n_windows": len(pick), "runtime": runtime_fingerprint(torch.device("cpu"))}
    (audit_args.out / "p300_paired_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows: list[dict] = []
    checks: list[dict] = []
    manifest: dict = {}
    started = time.time()
    for arch in architectures:
        for seed in seeds:
            checkpoint_dir = audit_args.reports / f"p300_{arch}_dynamics_true_seed{seed}_checkpoints"
            report = audit_args.reports / f"p300_{arch}_dynamics_true_seed{seed}.json"
            run_data = load_p300_cache(audit_args.cache, val_subjects="9,10", seed=seed, max_train_per_class=288, max_val_per_class=72)
            for epoch in epochs:
                if (arch, seed, epoch) not in available_keys:
                    continue
                checkpoint = checkpoint_dir / f"{arch}_true_epoch_{epoch:03d}.pt"
                shard = shard_path(audit_args.out, arch, seed, epoch)
                checkpoint_digest = sha256(checkpoint)
                names = FIXED_HOOKS[arch]
                shard_inputs = {"schema": "p300-paired-audit-shard-v2", "cache_sha256": cache_digest, "selected_cache_indices": pick.tolist(), "eps": audit_args.eps, "checkpoint_sha256": checkpoint_digest, "fixed_hooks": list(names), "batch_windows": audit_args.batch_windows, "runtime": runtime_fingerprint(torch.device("cpu"))}
                expected_keys = {(layer, int(pick[position])) for layer in names for position in range(len(pick))}
                saved = load_verified_shard(shard, fingerprint(shard_inputs), expected_rows=len(expected_keys), row_keys=expected_keys, row_key_fields=("layer", "cache_val_index"))
                if saved is not None:
                    rows.extend(saved["rows"])
                    checks.append(saved["check"])
                    manifest.setdefault(arch, {}).setdefault(str(seed), {})[str(epoch)] = saved.get("manifest", {})
                    continue
                model, layers, _ = arch_spec(arch, x.shape[1], x.shape[2], n_outputs=2, sfreq=data["sfreq"])
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
                model.eval()
                validation = model_accuracy(model, run_data["val_x"], run_data["val_y"])
                recorded = recorded_val_accuracy(report, epoch)
                if not _close_or_both_nan(validation, recorded, rtol=1e-6, atol=1e-6):
                    raise AssertionError((arch, seed, epoch, validation, recorded))
                unavailable_hooks = set(names) - set(layers) - {"logits"}
                if unavailable_hooks:
                    raise ValueError(f"fixed hooks unavailable for {arch}: {sorted(unavailable_hooks)}")
                capture = LayerCapture(model, names)
                try:
                    values = {name: [] for name in names}
                    shapes = {}
                    for offset in range(0, len(x), audit_args.batch_windows):
                        stop = min(offset + audit_args.batch_windows, len(x))
                        batch = components[offset:stop]
                        captured = capture(batch.reshape((-1,) + batch.shape[2:]), device=torch.device("cpu"))
                        zero = capture(np.zeros((1,) + batch.shape[2:], np.float32), device=torch.device("cpu"))
                        for name in names:
                            shapes[name] = list(captured[name].shape[1:])
                            feature_rows = captured[name].reshape(stop - offset, 5, -1).astype(np.float64)
                            baseline = zero[name][0].reshape(-1).astype(np.float64)
                            values[name].extend(paired_band_metrics(feature_rows, baseline, eps=audit_args.eps))
                    manifest.setdefault(arch, {}).setdefault(str(seed), {})[str(epoch)] = {"checkpoint_sha256": checkpoint_digest, "layers": list(names), "feature_shapes": shapes}
                    checkpoint_rows = []
                    for name in names:
                        for position, score in enumerate(values[name]):
                            source = int(data["val_idx"][pick[position]])
                            checkpoint_rows.append({"dataset": "p300", "arch": arch, "seed": seed, "epoch": epoch, "layer": name, "window_position": position, "cache_val_index": int(pick[position]), "subject": int(data["subject"][source]), "run": int(data["run"][source]), "event": int(data["event"][source]), "onset": int(data["onset"][source]), "label": int(y[position]), "checkpoint_sha256": checkpoint_digest, "recorded_val_accuracy": recorded, "recomputed_val_accuracy": validation, "feature_shape": json.dumps(shapes[name]), **score})
                finally:
                    capture.close()
                    del model
                    gc.collect()
                check = {"arch": arch, "seed": seed, "epoch": epoch, "recorded_val_accuracy": recorded, "recomputed_val_accuracy": validation, "abs_error": abs(validation - recorded)}
                write_verified_shard(shard, inputs=shard_inputs, rows=checkpoint_rows, check=check, manifest=manifest[arch][str(seed)][str(epoch)])
                rows.extend(checkpoint_rows)
                checks.append(check)
                print(arch, seed, epoch, len(rows), round(time.time() - started, 1), flush=True)
    if not rows or not checks:
        raise RuntimeError("no requested P300 checkpoints were available")
    with (audit_args.out / "p300_paired_raw_anchored.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (audit_args.out / "p300_checkpoint_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    (audit_args.out / "p300_layer_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "checkpoints": len(checks), "max_val_accuracy_error": max(item["abs_error"] for item in checks), "seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
