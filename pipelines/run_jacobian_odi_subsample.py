#!/usr/bin/env python3
"""Subsampled Jacobian-ODI audit.

For each selected epoch x and band component P_k x, this computes the local
linear response D Phi_x[P_k x] with torch JVPs, then compares the resulting
Jacobian-ODI against the ordinary band-probe ODI Phi(P_k x).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from scipy import stats

from hsdd.diagnostics import cosine_matrix, offdiag_mean_abs
from hsdd.synthetic import fft_band_components
from run_bci2a_hsdd_sensitivity import band_schemes
from run_cka_rsa_dynamics import checkpoint_file, default_checkpoint_dir
from run_cross_dataset_hsdd_sensitivity import load_dataset
from train_bci2a_arch_dynamics import compact_layers
from train_bci2a_arch_layer_audit import ARCHES, arch_spec
from train_eegnet_bci2a_layer_audit import LayerCapture, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bci2a", "sleepedf_full", "seediv", "p300"), default="bci2a")
    parser.add_argument("--arches", default="eegnet,shallowconvnet,eegconformer,tsception,atcnet")
    parser.add_argument("--label-mode", choices=("true", "shuffled"), default="true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epoch", default="final")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--band-scheme", default="canonical")
    parser.add_argument("--out-dir", default="reports/jacobian_odi")
    parser.add_argument("--bci-root", default=None)
    parser.add_argument("--sleep-cache", default="prepared/sleepedf_sc153_fpzcz_pzoz.npz")
    parser.add_argument("--sleep-val-recordings", type=int, default=31)
    parser.add_argument("--sleep-max-train-per-class", type=int, default=1600)
    parser.add_argument("--sleep-max-val-per-class", type=int, default=400)
    parser.add_argument("--seediv-cache", default="prepared/seediv_raw_4s_max8.npz")
    parser.add_argument("--seediv-val-subjects", default="13,14,15")
    parser.add_argument("--seediv-max-train-per-class", type=int, default=800)
    parser.add_argument("--seediv-max-val-per-class", type=int, default=200)
    parser.add_argument("--p300-cache", default="prepared/p300_bnci2014_009_raw_1s_200hz.npz")
    parser.add_argument("--p300-val-subjects", default="9,10")
    parser.add_argument("--p300-max-train-per-class", type=int, default=288)
    parser.add_argument("--p300-max-val-per-class", type=int, default=72)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arches(spec: str) -> list[str]:
    return [a.strip() for a in spec.replace(",", " ").split() if a.strip()]


def default_epochs(dataset: str) -> int:
    return 30 if dataset == "bci2a" else 20


def parse_epoch(spec: str, dataset: str) -> int:
    token = spec.strip().lower()
    if token in {"final", "last"}:
        return default_epochs(dataset)
    return int(token)


def available_layer_names(model: torch.nn.Module, layer_names: tuple[str, ...]) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    return tuple(name for name in compact_layers("", layer_names) if name == "logits" or name in modules)


def odi(features: np.ndarray) -> float:
    return offdiag_mean_abs(cosine_matrix(features.astype(np.float64)))


def layer_probe_outputs(
    model: torch.nn.Module,
    components: np.ndarray,
    layer: str,
    device: torch.device,
) -> np.ndarray:
    capture = LayerCapture(model.eval(), (layer,))
    try:
        outputs = capture(components.astype(np.float32), device=device)
    finally:
        capture.close()
    return outputs[layer].reshape(outputs[layer].shape[0], -1).astype(np.float64)


def layer_jvp_outputs(
    model: torch.nn.Module,
    layer: str,
    x: np.ndarray,
    directions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    modules = dict(model.named_modules())
    module = None if layer == "logits" else modules[layer]
    x_t = torch.from_numpy(x[None].astype(np.float32)).to(device)
    outs = []
    for direction in directions:
        v_t = torch.from_numpy(direction[None].astype(np.float32)).to(device)
        captured: dict[str, torch.Tensor] = {}
        handle = None
        if module is not None:
            handle = module.register_forward_hook(lambda _m, _inp, out: captured.__setitem__("z", out))

        def func(inp: torch.Tensor) -> torch.Tensor:
            captured.clear()
            y = model(inp)
            z = y if layer == "logits" else captured["z"]
            return z.reshape(-1)

        try:
            _y, jvp = torch.autograd.functional.jvp(func, (x_t,), (v_t,), create_graph=False, strict=False)
        finally:
            if handle is not None:
                handle.remove()
        outs.append(jvp.detach().cpu().numpy().reshape(-1))
    return np.asarray(outs, dtype=np.float64)


def corr(x: list[float], y: list[float], method: str) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(keep)) < 4:
        return float("nan"), float("nan")
    if method == "spearman":
        res = stats.spearmanr(a[keep], b[keep])
    else:
        res = stats.pearsonr(a[keep], b[keep])
    return float(res.statistic), float(res.pvalue)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = parse_epoch(args.epoch, args.dataset)
    data = load_dataset(args)
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(data["x"]), size=min(args.n_samples, len(data["x"])), replace=False)
    x_pool = data["x"][sample_idx].astype(np.float32)
    schemes = band_schemes((args.band_scheme,), data["train_x"], data["sfreq"])
    bands = schemes[args.band_scheme]
    rows = []

    for arch in parse_arches(args.arches):
        if args.dataset == "p300" and arch == "atcnet":
            continue
        model, all_layers, model_source = arch_spec(
            arch,
            data["train_x"].shape[1],
            data["train_x"].shape[2],
            n_outputs=data["n_outputs"],
            sfreq=data["sfreq"],
        )
        modules = dict(model.named_modules())
        layers = tuple(name for name in compact_layers(arch, all_layers) if name == "logits" or name in modules)
        ckpt_dir = default_checkpoint_dir(
            "sleepedf_full" if args.dataset == "sleepedf_full" else args.dataset,
            arch,
            args.label_mode,
            args.seed,
        )
        ckpt = checkpoint_file(ckpt_dir, arch, args.label_mode, epoch)
        if not ckpt.exists():
            print(f"skip missing checkpoint {ckpt}", flush=True)
            continue
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        for sample_i, x in zip(sample_idx.tolist(), x_pool):
            components, _ = fft_band_components(x, sfreq=data["sfreq"], bands=bands)
            for layer in layers:
                probe_z = layer_probe_outputs(model, components, layer, device)
                jvp_z = layer_jvp_outputs(model, layer, x, components, device)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "arch": arch,
                        "mode": args.label_mode,
                        "seed": args.seed,
                        "epoch": epoch,
                        "sample_index": sample_i,
                        "band_scheme": args.band_scheme,
                        "n_bands": len(bands),
                        "layer": layer,
                        "model_source": model_source,
                        "probe_odi": odi(probe_z),
                        "jacobian_odi": odi(jvp_z),
                        "probe_output_dim": int(probe_z.shape[1]),
                        "jvp_output_dim": int(jvp_z.shape[1]),
                    }
                )
                print(f"done arch={arch} sample={sample_i} layer={layer}", flush=True)

    out_dir = Path(args.out_dir)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["arch"], row["mode"])].append(row)
    summary = []
    for (dataset, arch, mode), vals in sorted(grouped.items()):
        pear_r, pear_p = corr([v["jacobian_odi"] for v in vals], [v["probe_odi"] for v in vals], "pearson")
        spear_r, spear_p = corr([v["jacobian_odi"] for v in vals], [v["probe_odi"] for v in vals], "spearman")
        summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "mode": mode,
                "n_layer_sample_rows": len(vals),
                "jacobian_probe_pearson_r": pear_r,
                "jacobian_probe_pearson_p": pear_p,
                "jacobian_probe_spearman_r": spear_r,
                "jacobian_probe_spearman_p": spear_p,
                "jacobian_odi_mean": float(np.mean([v["jacobian_odi"] for v in vals])),
                "probe_odi_mean": float(np.mean([v["probe_odi"] for v in vals])),
            }
        )
    if rows:
        pear_r, pear_p = corr([v["jacobian_odi"] for v in rows], [v["probe_odi"] for v in rows], "pearson")
        spear_r, spear_p = corr([v["jacobian_odi"] for v in rows], [v["probe_odi"] for v in rows], "spearman")
        summary.insert(
            0,
            {
                "dataset": args.dataset,
                "arch": "pooled_all_architectures",
                "mode": args.label_mode,
                "n_layer_sample_rows": len(rows),
                "jacobian_probe_pearson_r": pear_r,
                "jacobian_probe_pearson_p": pear_p,
                "jacobian_probe_spearman_r": spear_r,
                "jacobian_probe_spearman_p": spear_p,
                "jacobian_odi_mean": float(np.mean([v["jacobian_odi"] for v in rows])),
                "probe_odi_mean": float(np.mean([v["probe_odi"] for v in rows])),
            },
        )

    write_csv(out_dir / "jacobian_odi_rows.csv", rows)
    write_csv(out_dir / "jacobian_odi_summary.csv", summary)
    lines = [
        "# Jacobian-ODI Subsample Audit",
        "",
        "Jacobian-ODI uses JVPs `D Phi_x[P_k x]`; probe-ODI uses ordinary band-isolated responses `Phi(P_k x)`.",
        "",
        "| Dataset | Architecture | Rows | Pearson r | Spearman r | Mean Jacobian ODI | Mean probe ODI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['arch']}` | {row['n_layer_sample_rows']} | "
            f"{row['jacobian_probe_pearson_r']:.3f} | {row['jacobian_probe_spearman_r']:.3f} | "
            f"{row['jacobian_odi_mean']:.3f} | {row['probe_odi_mean']:.3f} |"
        )
    (out_dir / "jacobian_odi_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
