# Reproducibility map

| Analysis element | Archive artifact | Inputs | Reproduction status |
|---|---|---|---|
| Finite-dimensional HSRG estimators | `hsrg/geometry.py` | Synthetic arrays or user-accessible EEG | Runs in this archive |
| FFT band-component construction | `hsrg/synthetic.py` | Synthetic arrays | Runs in this archive |
| Reference controls | `scripts/run_synthetic_hsrg.py` | Deterministic synthetic data | Runs in this archive |
| Unit checks | `tests/test_geometry.py` | No external data | Runs in this archive |
| Aggregate outcomes | `results/` and `provenance/` | Compact CSV reports | Inspectable in this archive |
| Fixed-hook P300 paired sensitivity | `results/p300/`, `pipelines/run_p300_paired_audit.py` | Public cache and compatible local checkpoints for inference; compact seed means for summaries | Aggregate summaries reproduce in this archive; inference inputs are not distributed |
| Taxonomy sensitivity | `results/taxonomy/`, `scripts/taxonomy_exclude_dataset_sensitivity.py` | Compact predictions for bootstrap; trajectory records for a refit | Bootstrap reproduces in this archive |
| Analytic Gaussian reference | `results/gaussian/`, `scripts/reaggregate_analytic_gaussian_null.py` | 4,644 compact canonical records | Reproduces in this archive |
| Training and audit pipelines | `pipelines/` with `configs/paths.example.json` | Source datasets obtained under their original terms | Runs after local path configuration |

The archive provides the estimator, deterministic reference controls, compact aggregate records, and scripts that consume locally obtained source datasets. Raw EEG, trained checkpoints, credentials, and local host paths are not distributed. Historical empirical ODI fields use the unanchored procedure; `hsrg.geometry.anchored_band_orthogonality_report` implements the distinct zero-anchored estimator for a matched audit pool.
