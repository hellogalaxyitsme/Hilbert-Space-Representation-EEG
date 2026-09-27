# Hilbert-Space Representation Geometry in EEG Neural Networks

This release contains analysis code and compact aggregate records for a study of how EEG-network layers transform disjoint spectral components. It provides the primary raw Orthogonality Distortion Index (raw ODI), a zero-anchored paired sensitivity, fixed-fold taxonomy summaries, and an analytic Gaussian dimension reference.

## Included analyses

- `hsrg/geometry.py` implements raw ODI over every off-diagonal pair, using `max(norm_i, eps) * max(norm_j, eps)`.
- `hsrg/audit_metrics.py` provides the raw/anchored paired estimator and signed baseline, cross, and residual decomposition. The signed terms reconstruct signed raw cosine; absolute ODI is not their sum.
- `pipelines/run_p300_paired_audit.py` is the fixed-hook P300 audit. It uses 64 fixed class-balanced held-out windows and fixed named layers at every checkpoint.
- `scripts/taxonomy_exclude_dataset_sensitivity.py` refits leave-dataset-out predictions from locally available trajectory records. It validates inventory membership, mappings, duplicate keys, empty folds, and bootstrap inputs.
- `scripts/reproduce_taxonomy_bootstrap.py` reruns compact four-task and three-task paired bootstrap inputs without EEG or weights.
- `scripts/reaggregate_analytic_gaussian_null.py` aggregates the supplied 4,644 canonical records using raw ODI minus alpha_d, the analytic Gaussian expectation.

## Reproduce compact numerical summaries

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/summarize_p300_paired.py --input results/p300/fixed_hook_layer_seed.csv --out outputs/p300_fixed_hook_summary.csv
python scripts/reproduce_taxonomy_bootstrap.py --input results/taxonomy/four_task_predictions.csv --out outputs/four_task_bootstrap.csv
python scripts/reproduce_taxonomy_bootstrap.py --input results/taxonomy/three_task_excluding_sleep_predictions.csv --out outputs/three_task_bootstrap.csv
python scripts/taxonomy_exclude_dataset_sensitivity.py --joined results/taxonomy/joined_layer_checkpoint_metrics.csv --taxonomy results/taxonomy/architecture_taxonomy_layer_rows.csv --datasets bci2a,seediv,p300 --out outputs/refit_predictions.csv --summary outputs/refit_summary.csv --fold-table outputs/refit_folds.csv --draws 5000 --seed 20260927
python scripts/reaggregate_analytic_gaussian_null.py --input results/gaussian/analytic_gaussian_rows.csv --summary outputs/analytic_gaussian_summary.csv
python scripts/reproduce_regularizer_ci.py --input results/regularizer/tier3_arm4_vs_arm2_pairs.csv --out outputs/regularizer_ci.csv
python scripts/validate_seediv_audit.py --summary results/seediv/expanded_audit_summary.json --coverage results/seediv/checkpoint_coverage.csv --fixed-hooks results/seediv/fixed_hook_seed_layer_checkpoint_summary.csv
python scripts/summarize_seediv_expanded_audit.py --input results/seediv/fixed_hook_seed_layer_checkpoint_summary.csv --out outputs/seediv_summary.csv
# The cache must contain one non-overlapping 4 s (800-sample at 200 Hz) window per trial.
python pipelines/run_seediv_expanded_audit.py --cache path/to/seediv_audit_4s_onewindow.npz --checkpoints path/to/seediv_checkpoints --out outputs/seediv_expanded
```

The fixed four-task taxonomy analysis reports 0.853352 taxonomy accuracy and 0.598196 training-only-majority accuracy, with paired difference 0.255156 [0.186345, 0.328171] across 1,140 rows and 114 run clusters. The Sleep-excluded refit reports 0.823260 and 0.580391, difference 0.242869 [0.161586, 0.329444], across 840 rows and 84 clusters. The latter excludes Sleep from both fitting and evaluation; it does not remove the historical Sleep training/validation-recording overlap limitation.

The P300 paired sensitivity covers 142 of 144 expected architecture/seed/checkpoint combinations. EEGNet, ShallowConvNet, and EEG-Conformer have all six seeds at epochs 0, 1, 2, 5, 10, and 20. TSception lacks seed 46 at epochs 10 and 20. For the fixed EEG-Conformer `patch_embedding` hook, mean raw ODI changed from 0.943625 at epoch 0 to 0.598648 at epoch 20, while mean anchored ODI changed from 0.155103 to 0.268836; all six seed-level changes have opposite raw and anchored directions. `results/p300/exclusion_coverage.csv` records zero raw-floor and anchored-exclusion counts for the retained full P300 rows and for that hook.

For local P300 inference, pass `--skip-missing` when requesting the recorded 142-checkpoint scope so the two unavailable TSception seed-46 late checkpoints are recorded in `p300_checkpoint_coverage.json` before any inference starts. Each completed checkpoint is written to `checkpoint_shards/` and merged on a later invocation.

## Inputs, scope, and limits

The expanded SEED-IV audit evaluated 180 checkpoints across five architectures, six seeds, and epochs 0, 1, 2, 5, 10, and 20. It retained 34 fixed named hooks, yielding 1,224 seed/layer/checkpoint summaries from 78,336 window rows; the row-level input is not distributed. All raw-floor and anchored-exclusion counts were zero and the largest signed reconstruction error was `9.094947017729282e-13`. At EEG-Conformer `patch_embedding`, raw ODI changed from `0.9702841371276435` to `0.7114471819012422`, while anchored ODI changed from `0.09515090464100821` to `0.1645733005233228`. Raw ODI fell for all six seeds; anchored ODI rose for five, while seed 45 changed by `-0.006008108104795198`. These are observed audit differences and do not establish a training mechanism.

Compact aggregate records and reproducible analysis code are included. Re-running P300 or SEED-IV inference requires data obtained under the provider terms and compatible local checkpoints. The SEED-IV audit expects the documented 200 Hz, one 4-second window-per-trial cache, all 180 compatible checkpoint files in the documented directory layout, and a CUDA-capable PyTorch installation; those scientific inputs are not redistributed. Resume metadata binds each shard to the cache and checkpoint digests, selected windows, fixed hooks, batch setting, and CUDA/PyTorch runtime.

The regularizer records describe a feature-centered smooth surrogate with epsilon `1e-8`, batch size 4, application every fourth batch, HSRG lambda 0.05, weight orthogonality penalty 0.0001, band noise 0.25 per validation epoch, channel/time standard-deviation normalization, and dropout 0.25 under eight masks shared across validation epochs. The authoritative initial-sweep intervals are in `results/regularizer/authoritative_ci.csv`.

Historical Tier-2 artifact fields remain available for reproduction. Their high-dimensional `gaussian_projection_odi` branch returns the input ODI, and the routine named `orthogonal_projection_odi` uses a Rademacher sign sketch without QR factorization. `results/gaussian/null_family_definitions.csv` preserves their original meanings. The analytic Gaussian reference is the primary dimension-only comparison in this release.

The original project code is intentionally UNLICENSED. Runtime dependencies retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
