# Hilbert-Space Representation Geometry in EEG Neural Networks

Analysis code and compact numerical results for studying how the layers of EEG neural networks transform disjoint spectral components. Each EEG epoch is split into orthogonal Fourier bands, each band is passed through a trained network, and the geometry of the layer responses is summarized with the orthogonality distortion index (ODI), the mean absolute cosine between band responses.

The code provides:

- raw ODI, computed from band responses directly, and anchored ODI, computed after subtracting each layer's response to an all-zero input;
- the signed decomposition of the raw cosine into baseline, cross and residual terms;
- the analytic Gaussian dimension reference, $\alpha_d = \Gamma(d/2) / (\sqrt{\pi}\,\Gamma((d+1)/2))$;
- training and measurement pipelines for EEGNet, ShallowConvNet, EEG-Conformer, TSception and ATCNet on BCI Competition IV-2a, Sleep-EDF, SEED-IV and BNCI2014-009 (P300);
- label-control, cross-task prediction, representation-similarity and regularization analyses.

## Installation

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`environment.yml` and `environment_validated_freeze.txt` describe the tested environment.

## Repository layout

| Path | Contents |
|---|---|
| `hsrg/` | ODI estimators (`geometry.py`), the paired raw/anchored estimator with the signed decomposition (`audit_metrics.py`), feature capture and synthetic checks |
| `hsdd/` | Band decomposition utilities (`hsdd` is the earlier package name, kept for compatibility) |
| `pipelines/` | Training and measurement pipelines for each dataset and architecture |
| `anchored/` | Raw and zero-anchored measurements at representative layers, and the derived statistics |
| `scripts/` | Reproduction of compact summaries (cross-task prediction, Gaussian reference, regularizer) |
| `results/` | Compact numerical results; `results/anchored/` holds the representative-layer measurements |
| `tests/` | Unit tests, including numerical checks of the baseline bound and its symmetric counterpart |

## Zero-anchored band geometry

The representative-layer measurements evaluate six to eight fixed layers per architecture on a fixed class-balanced set of held-out windows, at initialization and after each saved training epoch, for networks trained with true and with shuffled labels from the same initialization.

```bash
# Measurements (require the prepared datasets and trained model parameters, which are not distributed)
python anchored/run_p300_label_control.py --cache path/to/p300.npz --reports path/to/reports \
    --label-mode true --out results/anchored/p300_true
python anchored/run_anchored_heldout.py --dataset bci2a --project path/to/project \
    --bci-root path/to/bci_iv_2a_fif --label-mode true --out results/anchored/bci2a_true
python anchored/run_anchored_heldout.py --dataset sleepedf_full --project path/to/project \
    --label-mode true --out results/anchored/sleepedf_full_true
# (repeat with --label-mode shuffled)

# Derived statistics from the included measurements (no EEG data or model parameters needed)
python anchored/analyses.py
```

`anchored/analyses.py` reads the measurements in `results/anchored/` and `results/seediv/`, writes summary statistics to `results/anchored/statistics.json` and derived tables to `outputs/anchored/`. It computes:

- baseline terms by architecture and task;
- the direction comparison between raw and anchored ODI, including a minimum-change sensitivity analysis;
- accuracy coupling and leave-one-task-out prediction of its sign;
- the true- versus shuffled-label comparison.

Bootstrap intervals resample trained networks within each task (5,000 draws, seed 2026).

## Compact summaries of the all-layer measurements

```bash
python scripts/summarize_p300_paired.py --input results/p300/fixed_hook_layer_seed.csv --out outputs/p300_fixed_hook_summary.csv
python scripts/reproduce_taxonomy_bootstrap.py --input results/taxonomy/four_task_predictions.csv --out outputs/four_task_bootstrap.csv
python scripts/reproduce_taxonomy_bootstrap.py --input results/taxonomy/three_task_excluding_sleep_predictions.csv --out outputs/three_task_bootstrap.csv
python scripts/taxonomy_exclude_dataset_sensitivity.py --joined results/taxonomy/joined_layer_checkpoint_metrics.csv \
    --taxonomy results/taxonomy/architecture_taxonomy_layer_rows.csv --datasets bci2a,seediv,p300 \
    --out outputs/refit_predictions.csv --summary outputs/refit_summary.csv --fold-table outputs/refit_folds.csv --draws 5000 --seed 20260927
python scripts/reaggregate_analytic_gaussian_null.py --input results/gaussian/analytic_gaussian_rows.csv --summary outputs/analytic_gaussian_summary.csv
python scripts/reproduce_regularizer_ci.py --input results/regularizer/tier3_arm4_vs_arm2_pairs.csv --out outputs/regularizer_ci.csv
python scripts/summarize_seediv_expanded_audit.py --input results/seediv/fixed_hook_seed_layer_checkpoint_summary.csv --out outputs/seediv_summary.csv
# SEED-IV representative-layer measurements (one non-overlapping 4-s window per held-out trial):
python pipelines/run_seediv_expanded_audit.py --cache path/to/seediv_4s_onewindow.npz --checkpoints path/to/seediv_models --out outputs/seediv_expanded
```

The regularizer analysis uses a feature-centred smooth surrogate (epsilon `1e-8`, regularizer batch of 4, applied every fourth step, weight 0.05) and compares architecture-conditioned targets with ODI minimization; paired intervals are in `results/regularizer/authoritative_ci.csv`.

## Data

The EEG datasets are publicly available from their original distributors under their original access terms: BCI Competition IV-2a, Sleep-EDF (PhysioNet), SEED-IV and BNCI2014-009. Raw EEG and trained model parameters are not redistributed. Rerunning the measurement pipelines requires the datasets and compatible trained models.

## License

The project code is unlicensed. Runtime dependencies retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
