#!/usr/bin/env python3
"""Resumable fixed-window SEED-IV raw/anchored checkpoint audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "pipelines"), str(CODE)]
from hsrg.checkpoint_shards import atomic_write_json, fingerprint

ARCHES = ("eegnet", "shallowconvnet", "eegconformer", "tsception", "atcnet")
SEEDS = tuple(range(41, 47))
EPOCHS = (0, 1, 2, 5, 10, 20)
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def runtime_fingerprint(device: torch.device) -> dict[str, object]:
    """Capture runtime settings that can alter floating-point inference."""
    cuda_available = torch.cuda.is_available()
    return {
        "schema": "seediv-expanded-audit-shard-v2",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if cuda_available else None,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32 if cuda_available else None,
        "device_name": torch.cuda.get_device_name(device) if cuda_available else None,
    }


def _finite_metric(value: str) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def valid_resume_rows(
    rows: list[dict[str, str]],
    *,
    arch: str,
    seed: int,
    epoch: int,
    names: tuple[str, ...],
    picked: np.ndarray,
    checkpoint_relpath: str,
    checkpoint_sha256: str,
) -> bool:
    """Verify exact shard identities and defined metric values before reuse."""
    expected = {(name, str(position)) for name in names for position in range(len(picked))}
    if len(rows) != len(expected):
        return False
    actual: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        identity = (row.get("layer", ""), row.get("window_position", ""))
        if identity in actual:
            return False
        actual.add(identity)
        if (
            row.get("arch") != arch
            or row.get("seed") != str(seed)
            or row.get("epoch") != str(epoch)
            or row.get("checkpoint_relpath") != checkpoint_relpath
            or row.get("checkpoint_sha256") != checkpoint_sha256
        ):
            return False
        try:
            position = int(row["window_position"])
        except (KeyError, TypeError, ValueError):
            return False
        if position < 0 or position >= len(picked) or row.get("cache_val_index") != str(int(picked[position])):
            return False
        try:
            active_bands = int(row["anchor_active_bands"])
            excluded_bands = int(row["anchor_excluded_bands"])
            excluded_pairs = int(row["anchor_excluded_pairs"])
        except (KeyError, TypeError, ValueError):
            return False
        total_bands = active_bands + excluded_bands
        expected_excluded_pairs = total_bands * (total_bands - 1) - active_bands * (active_bands - 1)
        if active_bands < 0 or excluded_bands < 0 or excluded_pairs != expected_excluded_pairs:
            return False
        has_active_pairs = active_bands >= 2
        for metric in ("raw_odi", "raw_signed_mean", "anchored_odi", "anchored_signed_mean"):
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                return False
            if metric.startswith("anchored_") and np.isnan(value):
                if has_active_pairs:
                    return False
            elif not np.isfinite(value):
                return False
        for metric in (
            "baseline_sq_over_raw_den",
            "cross_over_raw_den",
            "residual_dot_over_raw_den",
            "signed_reconstruction_max_abs_error",
            "response_norm_mean",
            "residual_norm_mean",
        ):
            if not _finite_metric(row.get(metric, "")):
                return False
    return actual == expected


def resume_metadata_matches(
    prior: object,
    shard: Path,
    rows: list[dict[str, str]],
    *,
    resume_inputs: dict,
    arch: str,
    seed: int,
    epoch: int,
    names: tuple[str, ...],
    picked: np.ndarray,
    checkpoint_relpath: str,
    checkpoint_sha256: str,
) -> bool:
    """Return whether metadata and CSV both describe the current checkpoint run."""
    if not isinstance(prior, dict):
        return False
    stored_inputs = prior.get("resume_inputs")
    stored_digest = prior.get("resume_fingerprint")
    if not isinstance(stored_inputs, dict) or not isinstance(stored_digest, str):
        return False
    if stored_digest != fingerprint(stored_inputs) or stored_digest != fingerprint(resume_inputs):
        return False
    if prior.get("csv_sha256") != sha256(shard):
        return False
    return valid_resume_rows(
        rows,
        arch=arch,
        seed=seed,
        epoch=epoch,
        names=names,
        picked=picked,
        checkpoint_relpath=checkpoint_relpath,
        checkpoint_sha256=checkpoint_sha256,
    )


def arrays_to_metrics(z: np.ndarray, baseline: np.ndarray, inputs: np.ndarray) -> list[dict]:
    """All-pair raw epsilon ODI plus separately computed signed components."""
    raw_norm = np.linalg.norm(z, axis=-1)
    residual = z - baseline[None, None, :]
    residual_norm = np.linalg.norm(residual, axis=-1)
    denominator = np.maximum(raw_norm[:, :, None], EPS) * np.maximum(raw_norm[:, None, :], EPS)
    raw_dot = np.einsum("wkd,wjd->wkj", z, z)
    raw_cos = raw_dot / denominator
    residual_unit = residual / np.maximum(residual_norm[..., None], EPS)
    anchor_cos = np.einsum("wkd,wjd->wkj", residual_unit, residual_unit)
    k = z.shape[1]
    offdiag = ~np.eye(k, dtype=bool)
    baseline_square = float(np.dot(baseline, baseline))
    baseline_residual = np.einsum("d,wkd->wk", baseline, residual)
    residual_dot = np.einsum("wkd,wjd->wkj", residual, residual)
    result: list[dict] = []
    for w in range(z.shape[0]):
        active = residual_norm[w] > EPS
        active_pairs = np.outer(active, active) & offdiag
        signed_baseline = baseline_square / denominator[w]
        signed_cross = (baseline_residual[w, :, None] + baseline_residual[w, None, :]) / denominator[w]
        signed_residual = residual_dot[w] / denominator[w]
        reconstructed = signed_baseline + signed_cross + signed_residual
        input_norm = np.linalg.norm(inputs[w], axis=(-2, -1))
        result.append({
            "raw_odi": float(np.abs(raw_cos[w][offdiag]).mean()),
            "raw_signed_mean": float(raw_cos[w][offdiag].mean()),
            "anchored_odi": float(np.abs(anchor_cos[w][active_pairs]).mean()) if active_pairs.any() else float("nan"),
            "anchored_signed_mean": float(anchor_cos[w][active_pairs].mean()) if active_pairs.any() else float("nan"),
            "raw_floor_bands": int((raw_norm[w] < EPS).sum()),
            "raw_floor_pairs": int((offdiag & ((raw_norm[w, :, None] < EPS) | (raw_norm[w, None, :] < EPS))).sum()),
            "anchor_active_bands": int(active.sum()),
            "anchor_excluded_bands": int(k - active.sum()),
            "anchor_excluded_pairs": int(offdiag.sum() - active_pairs.sum()),
            "baseline_sq_over_raw_den": float(signed_baseline[offdiag].mean()),
            "cross_over_raw_den": float(signed_cross[offdiag].mean()),
            "residual_dot_over_raw_den": float(signed_residual[offdiag].mean()),
            "signed_reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - raw_cos[w]))),
            "response_norm_mean": float(raw_norm[w].mean()),
            "residual_norm_mean": float(residual_norm[w].mean()),
            "input_band_norms": json.dumps(input_norm.tolist()),
            "response_band_norms": json.dumps(raw_norm[w].tolist()),
            "residual_band_norms": json.dumps(residual_norm[w].tolist()),
        })
    return result


def aliases(captured: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for position, name in enumerate(names):
        for previous in names[:position]:
            if captured[name].shape == captured[previous].shape and np.allclose(
                captured[name], captured[previous], rtol=2e-5, atol=2e-6, equal_nan=True
            ):
                found[name] = previous
                break
    return found


def cache_selection(cache: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    from seediv_utils import load_seediv_cache
    from hsdd.synthetic import fft_band_components
    data = load_seediv_cache(cache, val_subjects="13,14,15", seed=41, max_train_per_class=0, max_val_per_class=0)
    rng = np.random.default_rng(41)
    selected: list[int] = []
    for label in sorted(set(data["val_y"].tolist())):
        candidates = np.flatnonzero(data["val_y"] == label)
        if len(candidates) < 16:
            raise RuntimeError(f"label {label} has only {len(candidates)} validation windows")
        selected.extend(rng.choice(candidates, size=16, replace=False).tolist())
    picked = np.asarray(sorted(selected), dtype=int)
    x = data["val_x"][picked]
    components = np.stack([fft_band_components(epoch, sfreq=data["sfreq"])[0] for epoch in x]).astype(np.float32)
    return data, picked, x, components


def checkpoint_path(root: Path, arch: str, seed: int, epoch: int) -> Path:
    return root / f"seediv_{arch}_dynamics_true_seed{seed}_checkpoints" / f"{arch}_true_epoch_{epoch:03d}.pt"


def capture_batch(capture, batch: np.ndarray, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    w, k, channels, times = batch.shape
    outputs = capture(batch.reshape(w * k, channels, times), device=device)
    zero = capture(np.zeros((1, channels, times), dtype=np.float32), device=device)
    return ({name: value.reshape(w, k, -1).astype(np.float64) for name, value in outputs.items()},
            {name: value[0].reshape(-1).astype(np.float64) for name, value in zero.items()})


def run_one(arch: str, seed: int, epoch: int, *, checkpoint_root: Path, out: Path, data: dict, picked: np.ndarray, components: np.ndarray, batch_windows: int, device: torch.device, cache_sha256: str) -> dict:
    from train_bci2a_arch_layer_audit import arch_spec
    from train_bci2a_arch_dynamics import compact_layers
    from train_eegnet_bci2a_layer_audit import LayerCapture
    checkpoint = checkpoint_path(checkpoint_root, arch, seed, epoch)
    shard = out / "shards" / f"{arch}_seed{seed}_epoch{epoch:03d}.csv"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_digest = sha256(checkpoint)
    model, declared_layers, _ = arch_spec(arch, components.shape[2], components.shape[3], n_outputs=4, sfreq=data["sfreq"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    model.to(device).eval()
    names = compact_layers(arch, declared_layers)  # fixed historical named set
    checkpoint_relpath = str(checkpoint.relative_to(checkpoint_root))
    resume_inputs = {
        "schema": "seediv-expanded-audit-shard-v2",
        "cache_sha256": cache_sha256,
        "selected_indices": picked.tolist(),
        "eps": EPS,
        "checkpoint_relpath": checkpoint_relpath,
        "checkpoint_sha256": checkpoint_digest,
        "fixed_hooks": list(names),
        "batch_windows": batch_windows,
        "runtime": runtime_fingerprint(device),
    }
    resume_digest = fingerprint(resume_inputs)
    metadata_path = out / "metadata" / f"{arch}_seed{seed}_epoch{epoch:03d}.json"
    if shard.exists() and metadata_path.exists():
        try:
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            with shard.open(encoding="utf-8", newline="") as handle:
                prior_rows = list(csv.DictReader(handle))
            if resume_metadata_matches(
                prior,
                shard,
                prior_rows,
                resume_inputs=resume_inputs,
                arch=arch,
                seed=seed,
                epoch=epoch,
                names=names,
                picked=picked,
                checkpoint_relpath=checkpoint_relpath,
                checkpoint_sha256=checkpoint_digest,
            ):
                return {"status": "resumed", "shard": str(shard), "rows": len(prior_rows), "checkpoint": str(checkpoint)}
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError, ValueError):
            pass
    capture = LayerCapture(model, names)
    rows: list[dict] = []
    metadata: dict = {"checkpoint_sha256": checkpoint_digest, "fixed_named_hooks": list(names), "feature_shapes": {}, "identity_aliases_first_batch": {}}
    try:
        first_output = None
        for start in range(0, len(components), batch_windows):
            stop = min(start + batch_windows, len(components))
            output, baseline = capture_batch(capture, components[start:stop], device)
            if first_output is None:
                first_output = {name: value.reshape(value.shape[0] * value.shape[1], -1) for name, value in output.items()}
                metadata["identity_aliases_first_batch"] = aliases(first_output, names)
            for name in names:
                metadata["feature_shapes"][name] = list(output[name].shape[2:])
                metrics = arrays_to_metrics(output[name], baseline[name], components[start:stop])
                for offset, item in enumerate(metrics):
                    pos = start + offset
                    source = int(data["val_idx"][picked[pos]])
                    rows.append({
                        "dataset": "seediv", "arch": arch, "seed": seed, "epoch": epoch, "layer": name,
                        "window_position": pos, "cache_val_index": int(picked[pos]), "label": int(data["val_y"][picked[pos]]),
                        "subject": int(data["subject"][source]), "session": int(data["session"][source]),
                        "trial": int(data["trial"][source]), "window": int(data["window"][source]),
                        "checkpoint_relpath": checkpoint_relpath, "checkpoint_sha256": checkpoint_digest,
                        "feature_shape": json.dumps(metadata["feature_shapes"][name]),
                        "identity_alias_first_batch": metadata["identity_aliases_first_batch"].get(name, ""),
                        "aggregation_included": 1, **item,
                    })
    finally:
        capture.close()
        del model
        torch.cuda.empty_cache()
    atomic_csv(shard, rows)
    metadata["n_rows"] = len(rows)
    metadata["csv_sha256"] = sha256(shard)
    metadata["resume_fingerprint"] = resume_digest
    metadata["resume_inputs"] = resume_inputs
    atomic_write_json(metadata_path, metadata)
    return {"status": "computed", "shard": str(shard), "rows": len(rows), "checkpoint": str(checkpoint), "sha256": checkpoint_digest}


def validate_batch_single(arch: str, *, checkpoint_root: Path, data: dict, components: np.ndarray, device: torch.device) -> list[dict]:
    """Compare one-window and batched execution at two endpoints for each architecture."""
    from train_bci2a_arch_layer_audit import arch_spec
    from train_bci2a_arch_dynamics import compact_layers
    from train_eegnet_bci2a_layer_audit import LayerCapture
    checks: list[dict] = []
    for epoch in (0, 20):
        checkpoint = checkpoint_path(checkpoint_root, arch, 41, epoch)
        model, declared, _ = arch_spec(arch, components.shape[2], components.shape[3], n_outputs=4, sfreq=data["sfreq"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        model.to(device).eval()
        names = compact_layers(arch, declared)
        capture = LayerCapture(model, names)
        try:
            batched, batched_base = capture_batch(capture, components[:4], device)
            single, single_base = capture_batch(capture, components[:1], device)
            for name in names:
                left = arrays_to_metrics(batched[name][:1], batched_base[name], components[:1])[0]
                right = arrays_to_metrics(single[name], single_base[name], components[:1])[0]
                checks.append({"arch": arch, "seed": 41, "epoch": epoch, "layer": name,
                               "raw_abs_error": abs(left["raw_odi"] - right["raw_odi"]),
                               "anchor_abs_error": abs(left["anchored_odi"] - right["anchored_odi"]),
                               "signed_reconstruction_max_abs_error": max(left["signed_reconstruction_max_abs_error"], right["signed_reconstruction_max_abs_error"])})
        finally:
            capture.close()
            del model
            torch.cuda.empty_cache()
    return checks


def aggregate(out: Path) -> None:
    import pandas as pd
    shards = sorted((out / "shards").glob("*.csv"))
    if len(shards) != 180:
        raise RuntimeError(f"expected 180 shards, found {len(shards)}")
    frame = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
    frame.to_csv(out / "seediv_expanded_per_window.csv", index=False)
    metrics = ["raw_odi", "anchored_odi", "raw_signed_mean", "anchored_signed_mean", "baseline_sq_over_raw_den", "cross_over_raw_den", "residual_dot_over_raw_den", "raw_floor_bands", "raw_floor_pairs", "anchor_excluded_bands", "anchor_excluded_pairs", "signed_reconstruction_max_abs_error"]
    by_layer = frame.groupby(["arch", "seed", "epoch", "layer"], as_index=False)[metrics].mean()
    by_layer["delta_anchor_minus_raw"] = by_layer["anchored_odi"] - by_layer["raw_odi"]
    by_layer.to_csv(out / "seediv_fixed_hook_seed_layer_checkpoint_summary.csv", index=False)
    by_arch = by_layer.groupby(["arch", "epoch"], as_index=False).agg({**{m: ["mean", "std"] for m in metrics}, "delta_anchor_minus_raw": ["mean", "std"], "layer": "nunique", "seed": "nunique"})
    by_arch.columns = ["_".join(v).strip("_") for v in by_arch.columns]
    by_arch.to_csv(out / "seediv_architecture_checkpoint_summary.csv", index=False)
    patch = by_layer[(by_layer.arch == "eegconformer") & (by_layer.layer == "patch_embedding") & (by_layer.epoch.isin([0, 20]))]
    patch.to_csv(out / "seediv_eegconformer_patch_endpoints.csv", index=False)
    coverage = frame.groupby(["arch", "seed", "epoch"], as_index=False).agg(n_rows=("layer", "size"), n_layers=("layer", "nunique"), n_windows=("window_position", "nunique"), checkpoint_sha256=("checkpoint_sha256", "first"))
    coverage.to_csv(out / "seediv_checkpoint_coverage.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-windows", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this audit")
    data, picked, _x, components = cache_selection(args.cache)
    if len(picked) != 64 or [int((data["val_y"][picked] == k).sum()) for k in range(4)] != [16] * 4:
        raise AssertionError("fixed 64-window class-balanced selection failed")
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {
        "cache_sha256": sha256(args.cache),
        "cache_filename": args.cache.name,
        "sfreq": data["sfreq"],
        "cache_metadata": data["cache_meta"],
        "val_pool_size": int(len(data["val_y"])),
        "val_pool_label_counts": {str(label): int((data["val_y"] == label).sum()) for label in sorted(set(data["val_y"].tolist()))},
        "selection_rng": "numpy.default_rng(41), per label choice(16, replace=False), then ascending validation index",
        "selected_val_indices": picked.tolist(),
        "labels": data["val_y"][picked].astype(int).tolist(),
        "subjects": data["subject"][data["val_idx"][picked]].astype(int).tolist(),
        "sessions": data["session"][data["val_idx"][picked]].astype(int).tolist(),
        "trials": data["trial"][data["val_idx"][picked]].astype(int).tolist(),
        "windows": data["window"][data["val_idx"][picked]].astype(int).tolist(),
        "eps": EPS,
        "raw_denominator": "max(norm_i,1e-12)*max(norm_j,1e-12)",
        "anchored_active_rule": "residual norm > 1e-12",
        "fixed_hook_rule": "historical compact named hooks retained for every checkpoint; aliases metadata only",
    }
    (args.out / "seediv_fixed64_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    device = torch.device("cuda")
    all_paths = [checkpoint_path(args.checkpoints, arch, seed, epoch) for arch in ARCHES for seed in SEEDS for epoch in EPOCHS]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + "; ".join(missing))
    preflight = [{"checkpoint_relpath": str(path.relative_to(args.checkpoints)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in all_paths]
    (args.out / "seediv_checkpoint_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    sanity: list[dict] = []
    for arch in ARCHES:
        sanity.extend(validate_batch_single(arch, checkpoint_root=args.checkpoints, data=data, components=components, device=device))
    atomic_csv(args.out / "seediv_batched_vs_single.csv", sanity)
    maximum = max(max(row["raw_abs_error"], row["anchor_abs_error"]) for row in sanity)
    # Captured convolutional feature maps may differ slightly between batch sizes
    # on CUDA float32 kernels.  This threshold is set above the observed
    # endpoint discrepancy and is recorded with the output rather than treated
    # as exact arithmetic equivalence.
    if maximum > 2e-4:
        raise AssertionError(f"batched/single difference {maximum} exceeds float32 tolerance")
    started = time.time()
    manifest: list[dict] = []
    for arch in ARCHES:
        for seed in SEEDS:
            for epoch in EPOCHS:
                item = run_one(arch, seed, epoch, checkpoint_root=args.checkpoints, out=args.out, data=data, picked=picked, components=components, batch_windows=args.batch_windows, device=device, cache_sha256=selection["cache_sha256"])
                manifest.append(item)
                print(f"{arch} seed={seed} epoch={epoch} {item['status']} rows={item['rows']} elapsed={time.time()-started:.1f}s", flush=True)
    (args.out / "seediv_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    aggregate(args.out)
    regression = {"not_run": "No comparison input is accepted by this portable inference command."}
    (args.out / "seediv_completion.json").write_text(json.dumps({"completed": True, "n_checkpoints": len(manifest), "batch_single_max_abs_error": maximum, "seed41_regression": regression, "elapsed_seconds": time.time() - started}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "n_checkpoints": len(manifest), "batch_single_max_abs_error": maximum, "seed41_regression": regression}, indent=2), flush=True)


if __name__ == "__main__":
    main()
