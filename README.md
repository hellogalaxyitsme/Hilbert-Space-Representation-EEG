# Hilbert-Space Representation Geometry in EEG Neural Networks

This project uses Hilbert-space geometry to study how neural-network layers transform orthogonal spectral components of EEG. By decomposing each input into disjoint frequency bands and tracking their representations, it measures changes in component alignment, pairwise distances, and layerwise geometry during training.

The analysis covers EEGNet, ShallowConvNet, EEG-Conformer, TSception, and ATCNet across motor imagery (BCI IV-2a), sleep staging (Sleep-EDF), emotion recognition (SEED-IV), and P300 decoding. It combines the Orthogonality Distortion Index (ODI), controlled reference experiments, and comparisons with centered kernel alignment (CKA) and representational similarity analysis (RSA) to characterize architecture-dependent representation behavior.

## What is included

- Finite-dimensional HSRG estimators, deterministic controls, and geometry unit tests.
- Training, evaluation, and aggregation pipelines for the archived EEG analyses.
- Compact aggregate result tables and provenance records.
- A bounded paired SEED-IV checkpoint audit that compares the historical unanchored measure with the separate zero-anchored measure.
- Dependency files for installation and the exact eight-package environment record used for the clean lab validation.

## Quick start

```powershell
python -m pip install -r requirements.txt
python scripts/run_synthetic_hsrg.py --out outputs/synthetic_reference.json
python -m unittest discover -s tests -v
python scripts/smoke_five_architectures.py
python scripts/cluster_taxonomy_bootstrap.py --input results/taxonomy_oos_sign_predictions.csv --out outputs/cluster_bootstrap.csv
```

The synthetic runner is deterministic for a fixed seed. It exercises identity, linear, ReLU, and deliberate band-mixing maps and reports scale-adjusted pairwise geometry and component coherence. The five-architecture command performs CPU forward and one-step optimizer checks. The bootstrap command uses the packaged out-of-sample prediction rows and writes a dataset-stratified, run-cluster summary.

## Measures and interpretation

`hsrg.geometry` provides `pairwise_isometry_report`, `band_orthogonality_report`, and `local_directional_distortion`. A sampled epoch is treated as a finite-dimensional approximation to an EEG signal, with `dt` controlling Riemann-sum norm scaling.

Historical aggregate rows use the unanchored ODI procedure: isolated band responses are compared directly with an epsilon norm denominator. `hsrg.geometry.anchored_band_orthogonality_report` is a distinct estimator that subtracts the zero-input response and records near-zero exclusions. Do not combine the two estimators in a single historical trajectory.

HSRG is a descriptive representation measure, not a universal quality score or evidence that one geometry is optimal for every task.

## Paired SEED-IV checkpoint audit

`pipelines/audit_seediv_checkpoints.py` reproduces the bounded paired audit when supplied with a compatible locally prepared SEED-IV cache and saved `seediv_*_dynamics_true_seed41_checkpoints/*epoch_0[02]0.pt` files. The script fixes seed 41 and selects up to 16 held-out windows from each of four classes, for at most 64 windows. It evaluates epoch 0 and epoch 20 on CPU or CUDA and writes raw ODI, anchored ODI, their difference, near-zero rate, and selected indices.

```powershell
python pipelines/audit_seediv_checkpoints.py --cache prepared/seediv_audit_4s_onewindow.npz --checkpoints . --out outputs/seediv_paired_raw_anchored.csv --device cpu
```

`results/seediv_paired_raw_anchored_seed41_64.csv` is the exact 64-window, 68-summary output used for the paired sensitivity. The study is limited to the available SEED-IV seed-41 checkpoints; it does not generalize an anchored pattern to the other tasks.

## Real-data pipelines and access

Obtain each source dataset under its distributor terms. The available cache-preparation entry points are `pipelines/prepare_p300_cache.py`, `pipelines/prepare_seediv_cache.py`, and `pipelines/prepare_sleepedf_cache.py`; BCI IV-2a is loaded by its training scripts. Supervised architecture audits are exposed through `train_bci2a_arch_layer_audit.py`, `train_p300_arch_layer_audit.py`, `train_seediv_arch_layer_audit.py`, and `train_sleepedf_arch_layer_audit.py`. Dataset locations are supplied through command-line arguments; `configs/paths.example.json` is illustrative and is not read automatically.

Raw EEG, trained weights, credentials, local host paths, article sources, and figure-production scripts are not included. The original project code is intentionally UNLICENSED. Third-party components are not redistributed beyond their documented runtime dependencies. See `docs/REPRODUCIBILITY_MAP.md` for the available artifacts and reproduction limits.
