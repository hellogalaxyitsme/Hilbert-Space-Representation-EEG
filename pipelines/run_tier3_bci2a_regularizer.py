#!/usr/bin/env python3
"""Tier-3 BCI IV 2a representation-regularizer ablation.

This runner trains one architecture/seed/arm and evaluates robustness endpoints.
The HSDD arm uses a differentiable ODI surrogate on band-isolated inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsdd.io import write_json
from hsdd.synthetic import DEFAULT_BANDS
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec, effective_batch_size
from train_eegnet_bci2a_layer_audit import evaluate, label_counts, load_bci2a, make_train_val, set_seed


ARMS = ("none", "odi_min", "weight_orth", "taxon_hsdd")


TAXONOMY_TARGETS: dict[str, dict[str, float]] = {
    "eegnet": {
        "conv_temporal": 0.08,
        "conv_spatial": 0.18,
        "bnorm_1": 0.18,
        "conv_separable_depth": 0.22,
        "bnorm_2": 0.16,
        "pool_2": 0.14,
        "logits": 0.05,
    },
    "shallowconvnet": {
        "conv_time_spat": 0.12,
        "bnorm": 0.12,
        "conv_nonlin_exp": 0.25,
        "pool": 0.22,
        "pool_nonlin_exp": 0.25,
        "logits": 0.08,
    },
    "eegconformer": {
        "patch_embedding": 0.12,
        "transformer.0": 0.08,
        "transformer.2": 0.07,
        "transformer.5": 0.06,
        "fc": 0.05,
        "logits": 0.05,
    },
    "tsception": {
        "temporal_blocks.0": 0.18,
        "temporal_blocks.2": 0.18,
        "batch_temporal_lay": 0.16,
        "spatial_block_1": 0.10,
        "spatial_block_2": 0.10,
        "batch_spatial_lay": 0.08,
        "dense_layer": 0.06,
        "logits": 0.05,
    },
    "atcnet": {
        "conv_block": 0.20,
        "attention_blocks.0": 0.08,
        "attention_blocks.4": 0.08,
        "temporal_conv_nets.0": 0.08,
        "temporal_conv_nets.4": 0.08,
        "final_layer.4": 0.05,
        "logits": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-hsdd", type=float, default=0.05)
    parser.add_argument("--lambda-weight-orth", type=float, default=1e-4)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--target-scheme", default="hand_taxon")
    parser.add_argument("--reg-every", type=int, default=1)
    parser.add_argument("--reg-batch-size", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--channel-drop-p", type=float, default=0.25)
    parser.add_argument("--channel-drop-repeats", type=int, default=8)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def load_target_map(path: str | None, arch: str) -> tuple[dict[str, float], dict[str, Any]]:
    if not path:
        return dict(TAXONOMY_TARGETS.get(arch, {})), {
            "scheme": "hand_taxon",
            "source": "built_in",
        }
    target_path = Path(path)
    payload = json.loads(target_path.read_text())
    targets = payload.get("targets", payload)
    if arch not in targets:
        raise KeyError(f"target file {target_path} does not contain arch={arch}")
    metadata = payload.get("metadata", {})
    metadata = {**metadata, "source": str(target_path)}
    return {str(k): float(v) for k, v in targets[arch].items()}, metadata


class GradLayerCapture:
    def __init__(self, model: nn.Module, layer_names: tuple[str, ...]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.outputs: dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        self.handles = []
        for name in layer_names:
            if name == "logits":
                continue
            if name not in modules:
                raise KeyError(f"missing layer {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.outputs = {}
        logits = self.model(x)
        outputs = dict(self.outputs)
        outputs["logits"] = logits
        return outputs

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def torch_band_components(
    x: torch.Tensor,
    *,
    sfreq: float,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq).to(x.device)
    spectrum = torch.fft.rfft(x, dim=-1)
    comps = []
    names = []
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        filtered = torch.zeros_like(spectrum)
        filtered[..., mask] = spectrum[..., mask]
        comps.append(torch.fft.irfft(filtered, n=x.shape[-1], dim=-1))
        names.append(name)
    return torch.stack(comps, dim=1), tuple(names)


def smooth_odi_from_features(features: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if features.ndim < 3:
        raise ValueError(f"expected (batch, bands, ...) features, got {tuple(features.shape)}")
    batch, n_bands = features.shape[:2]
    flat = features.reshape(batch, n_bands, -1)
    flat = flat - flat.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(eps)
    unit = flat / norm
    gram = torch.matmul(unit, unit.transpose(1, 2))
    mask = ~torch.eye(n_bands, dtype=torch.bool, device=features.device)
    smooth_abs = torch.sqrt(gram[:, mask].pow(2) + eps)
    return smooth_abs.mean()


def hsdd_regularizer(
    model: nn.Module,
    xb: torch.Tensor,
    *,
    arch: str,
    arm: str,
    layer_names: tuple[str, ...],
    sfreq: float,
    target_map: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if arm not in {"odi_min", "taxon_hsdd"}:
        return xb.new_tensor(0.0), {}

    reg_x = xb[: max(1, min(len(xb), xb.shape[0]))]
    components, _ = torch_band_components(reg_x, sfreq=sfreq)
    batch, n_bands = components.shape[:2]
    band_batch = components.reshape(batch * n_bands, *components.shape[2:])

    capture = GradLayerCapture(model, layer_names)
    try:
        outputs = capture.forward(band_batch)
    finally:
        capture.close()

    active_target_map = (
        (target_map or TAXONOMY_TARGETS[arch])
        if arm == "taxon_hsdd"
        else {name: 0.0 for name in layer_names}
    )
    losses = []
    metrics: dict[str, float] = {}
    for name in layer_names:
        if name not in outputs:
            continue
        z = outputs[name].reshape(batch, n_bands, *outputs[name].shape[1:])
        odi = smooth_odi_from_features(z)
        target = xb.new_tensor(float(active_target_map.get(name, 0.0)))
        losses.append((odi - target).pow(2))
        metrics[f"{name}.odi_surrogate"] = float(odi.detach().cpu())
        metrics[f"{name}.target"] = float(target.detach().cpu())
    if not losses:
        return xb.new_tensor(0.0), metrics
    return torch.stack(losses).mean(), metrics


def weight_orthogonality_regularizer(model: nn.Module) -> torch.Tensor:
    losses = []
    for param in model.parameters():
        if param.ndim < 2:
            continue
        w = param.reshape(param.shape[0], -1)
        if w.shape[0] <= w.shape[1]:
            gram = w @ w.t()
            ident = torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
        else:
            gram = w.t() @ w
            ident = torch.eye(w.shape[1], device=w.device, dtype=w.dtype)
        losses.append((gram - ident).pow(2).mean())
    if not losses:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(losses).mean()


def stratified_fraction(x: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 0.999:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        n = max(1, int(round(len(idx) * fraction)))
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    keep_arr = np.asarray(sorted(keep), dtype=int)
    return x[keep_arr], y[keep_arr]


def train_one_epoch_tier3(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    arch: str,
    arm: str,
    hsdd_layers: tuple[str, ...],
    sfreq: float,
    lambda_hsdd: float,
    lambda_weight_orth: float,
    reg_every: int,
    target_map: dict[str, float] | None,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total = 0
    last_reg_metrics: dict[str, float] = {}
    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        task_loss = loss_fn(logits, yb)
        reg_loss = xb.new_tensor(0.0)
        if arm in {"odi_min", "taxon_hsdd"} and step % max(reg_every, 1) == 0:
            reg_x = xb[: min(len(xb), max(1, int(getattr(loader, "reg_batch_size", len(xb)))))]
            hsdd_loss, last_reg_metrics = hsdd_regularizer(
                model,
                reg_x,
                arch=arch,
                arm=arm,
                layer_names=hsdd_layers,
                sfreq=sfreq,
                target_map=target_map,
            )
            reg_loss = lambda_hsdd * hsdd_loss
        elif arm == "weight_orth":
            reg_loss = lambda_weight_orth * weight_orthogonality_regularizer(model)
        loss = task_loss + reg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(xb)
        total_task_loss += float(task_loss.detach().cpu()) * len(xb)
        total_reg_loss += float(reg_loss.detach().cpu()) * len(xb)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return {
        "train_loss": total_loss / total,
        "train_task_loss": total_task_loss / total,
        "train_regularizer_loss": total_reg_loss / total,
        "train_acc": total_correct / total,
        "last_regularizer_metrics": last_reg_metrics,
    }


@torch.no_grad()
def predict_arrays(model: nn.Module, x: np.ndarray, y: np.ndarray, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, float, float]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    probs = []
    labels = []
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item()) * len(xb)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += len(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(yb.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), total_loss / total, total_correct / total


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, *, n_bins: int) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def fft_band_limited_noise(shape: tuple[int, int, int], *, sfreq: float, band: tuple[str, float, float], rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=shape).astype(np.float32)
    freqs = np.fft.rfftfreq(shape[-1], d=1.0 / sfreq)
    spectrum = np.fft.rfft(noise, axis=-1)
    mask = (freqs >= band[1]) & (freqs < band[2])
    filtered = np.zeros_like(spectrum)
    filtered[..., mask] = spectrum[..., mask]
    out = np.fft.irfft(filtered, n=shape[-1], axis=-1).astype(np.float32)
    scale = np.std(out, axis=(-2, -1), keepdims=True)
    return out / np.maximum(scale, 1e-6)


def robustness_endpoints(
    model: nn.Module,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    sfreq: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    noise_scale: float,
    channel_drop_p: float,
    channel_drop_repeats: int,
    ece_bins: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1701)
    clean_probs, clean_labels, clean_loss, clean_acc = predict_arrays(model, val_x, val_y, batch_size=batch_size, device=device)
    clean_ece = expected_calibration_error(clean_probs, clean_labels, n_bins=ece_bins)
    x_std = np.std(val_x, axis=(-2, -1), keepdims=True).astype(np.float32)

    band_rows = []
    for band in DEFAULT_BANDS:
        noise = fft_band_limited_noise(val_x.shape, sfreq=sfreq, band=band, rng=rng)
        perturbed = val_x + noise_scale * x_std * noise
        probs, labels, loss, acc = predict_arrays(model, perturbed.astype(np.float32), val_y, batch_size=batch_size, device=device)
        band_rows.append(
            {
                "band": band[0],
                "low_hz": band[1],
                "high_hz": band[2],
                "loss": loss,
                "acc": acc,
                "ece": expected_calibration_error(probs, labels, n_bins=ece_bins),
                "acc_drop": clean_acc - acc,
            }
        )

    drop_rows = []
    for repeat in range(channel_drop_repeats):
        keep = rng.random(val_x.shape[1]) >= channel_drop_p
        if keep.all():
            keep[rng.integers(0, val_x.shape[1])] = False
        if not keep.any():
            keep[rng.integers(0, val_x.shape[1])] = True
        dropped = val_x.copy()
        dropped[:, ~keep, :] = 0.0
        probs, labels, loss, acc = predict_arrays(model, dropped.astype(np.float32), val_y, batch_size=batch_size, device=device)
        drop_rows.append(
            {
                "repeat": repeat,
                "n_dropped": int((~keep).sum()),
                "drop_fraction": float((~keep).mean()),
                "loss": loss,
                "acc": acc,
                "ece": expected_calibration_error(probs, labels, n_bins=ece_bins),
                "acc_drop": clean_acc - acc,
            }
        )

    band_acc = np.asarray([r["acc"] for r in band_rows], dtype=float)
    band_drop = np.asarray([r["acc_drop"] for r in band_rows], dtype=float)
    channel_acc = np.asarray([r["acc"] for r in drop_rows], dtype=float)
    channel_drop = np.asarray([r["acc_drop"] for r in drop_rows], dtype=float)
    return {
        "clean": {"loss": clean_loss, "acc": clean_acc, "ece": clean_ece},
        "band_limited_noise": {
            "noise_scale": noise_scale,
            "rows": band_rows,
            "mean_acc": float(band_acc.mean()),
            "mean_acc_drop": float(band_drop.mean()),
            "worst_acc": float(band_acc.min()),
            "worst_acc_drop": float(band_drop.max()),
        },
        "channel_dropout": {
            "drop_p": channel_drop_p,
            "repeats": channel_drop_repeats,
            "rows": drop_rows,
            "mean_acc": float(channel_acc.mean()),
            "mean_acc_drop": float(channel_drop.mean()),
            "worst_acc": float(channel_acc.min()),
            "worst_acc_drop": float(channel_drop.max()),
        },
    }


def epochs_to_fraction_of_best(history: list[dict[str, Any]], fraction: float = 0.9) -> int | None:
    best = max(float(row["val_acc"]) for row in history)
    threshold = fraction * best
    for row in history:
        if float(row["val_acc"]) >= threshold:
            return int(row["epoch"])
    return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bci2a(Path(args.root), tmin=args.tmin, tmax=args.tmax)
    train_x, train_y, val_x, val_y = make_train_val(data, label_mode="true", seed=args.seed)
    train_x, train_y = stratified_fraction(train_x, train_y, args.train_fraction, args.seed)
    sfreq = 1.0 / data["dt"]

    model, layer_names, model_source = arch_spec(args.arch, train_x.shape[1], train_x.shape[2], n_outputs=4, sfreq=sfreq)
    model.to(device)
    batch_size = effective_batch_size(args.arch, args.batch_size)
    hsdd_layers = compact_layers(args.arch, layer_names)
    target_map, target_metadata = load_target_map(args.target_file, args.arch)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    setattr(train_loader, "reg_batch_size", args.reg_batch_size)
    train_eval_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    best_val_acc = -math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_tier3(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            arch=args.arch,
            arm=args.arm,
            hsdd_layers=hsdd_layers,
            sfreq=sfreq,
            lambda_hsdd=args.lambda_hsdd,
            lambda_weight_orth=args.lambda_weight_orth,
            reg_every=args.reg_every,
            target_map=target_map,
        )
        train_eval_loss, train_eval_acc = evaluate(model, train_eval_loader, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            **train_metrics,
            "train_eval_loss": train_eval_loss,
            "train_eval_acc": train_eval_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} arch={args.arch} arm={args.arm} seed={args.seed} "
            f"train_acc={train_eval_acc:.3f} val_acc={val_acc:.3f} "
            f"reg={row['train_regularizer_loss']:.5f} best_loss={best_val_loss:.4f}@{best_epoch}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    endpoints = robustness_endpoints(
        model,
        val_x,
        val_y,
        sfreq=sfreq,
        batch_size=batch_size,
        device=device,
        seed=args.seed,
        noise_scale=args.noise_scale,
        channel_drop_p=args.channel_drop_p,
        channel_drop_repeats=args.channel_drop_repeats,
        ece_bins=args.ece_bins,
    )
    checkpoint = Path(
        args.checkpoint
        or f"reports/tier3_regularizer_bci2a/checkpoints/{args.arch}_{args.arm}_seed{args.seed}_frac{args.train_fraction:g}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "args": vars(args),
            "best_epoch_by_val_loss": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc_at_best_loss": best_val_acc,
        },
        checkpoint,
    )
    payload = {
        "dataset": {
            "name": "bci2a",
            "protocol": "cross-subject: A01T-A08T train, A09T validation/test proxy",
            "root": args.root,
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "train_label_counts": label_counts(train_y),
            "val_label_counts": label_counts(val_y),
            "sfreq": sfreq,
            "train_fraction": args.train_fraction,
        },
        "model": {
            "arch": args.arch,
            "source": model_source,
            "layer_names": layer_names,
            "hsdd_regularized_layers": hsdd_layers,
            "taxonomy_targets": target_map,
            "target_metadata": target_metadata,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "run": {
            "arm": args.arm,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_hsdd": args.lambda_hsdd,
            "lambda_weight_orth": args.lambda_weight_orth,
            "target_file": args.target_file,
            "target_scheme": args.target_scheme,
            "reg_every": args.reg_every,
            "reg_batch_size": args.reg_batch_size,
            "checkpoint": str(checkpoint),
        },
        "selection": {
            "criterion": "minimum clean validation loss; robustness endpoints are not used for model selection",
            "best_epoch_by_val_loss": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc_at_best_loss": best_val_acc,
            "epochs_to_90pct_run_best_val_acc": epochs_to_fraction_of_best(history, 0.9),
        },
        "history": history,
        "endpoints": endpoints,
    }
    out = Path(args.out or f"reports/tier3_regularizer_bci2a/{args.arch}_{args.arm}_seed{args.seed}_frac{args.train_fraction:g}.json")
    write_json(out, payload)
    print("wrote", out, flush=True)
    print(
        "summary",
        f"clean_acc={endpoints['clean']['acc']:.3f}",
        f"band_noise_drop={endpoints['band_limited_noise']['mean_acc_drop']:.3f}",
        f"channel_drop={endpoints['channel_dropout']['mean_acc_drop']:.3f}",
        f"ece={endpoints['clean']['ece']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
