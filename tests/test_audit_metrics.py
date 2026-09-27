import unittest

import numpy as np

from hsrg.audit_metrics import paired_band_metrics


class PairedBandMetricsTests(unittest.TestCase):
    def test_individual_norm_floors_and_signed_reconstruction(self):
        z = np.array([[[0.0, 0.0], [1e-15, 0.0], [2.0, 0.0]]])
        metric = paired_band_metrics(z, np.array([1.0, 0.0]), eps=1e-12)[0]
        self.assertEqual(metric["raw_floor_bands"], 2)
        # Tiny raw responses can have a large absolute cancellation error.
        # Its scale-aware rounding bound must still cover the observed error.
        self.assertLessEqual(metric["decomposition_max_abs_error"], metric["decomposition_error_bound"])

    def test_baseline_cancellation_is_signed_not_absolute_sum(self):
        baseline = np.array([2.0, 0.0])
        z = np.array([[[1.0, 0.0], [3.0, 0.0]]])
        metric = paired_band_metrics(z, baseline)[0]
        self.assertAlmostEqual(metric["raw_signed_mean"], 1.0)
        self.assertAlmostEqual(metric["baseline_sq_over_raw_den"] + metric["cross_over_raw_den"] + metric["residual_over_raw_den"], 1.0)
        self.assertAlmostEqual(metric["raw_odi"], 1.0)

    def test_anchored_excludes_zero_residual(self):
        z = np.array([[[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        metric = paired_band_metrics(z, np.array([1.0, 0.0]))[0]
        self.assertEqual(metric["excluded_bands"], 2)
        self.assertTrue(np.isnan(metric["anchored_odi"]))

    def test_direct_residual_term_matches_longdouble_reference(self):
        baseline = np.array([1.5, -2.0])
        responses = np.array([[[2.0, -1.0], [1.0, 3.0]]])
        metric = paired_band_metrics(responses, baseline)[0]
        residual = responses.astype(np.longdouble) - baseline.astype(np.longdouble)
        raw_norm = np.linalg.norm(responses.astype(np.longdouble), axis=-1)
        denominator = raw_norm[:, :, None] * raw_norm[:, None, :]
        expected = np.mean((residual @ np.swapaxes(residual, 1, 2))[0][~np.eye(2, dtype=bool)] / denominator[0][~np.eye(2, dtype=bool)])
        self.assertAlmostEqual(metric["residual_over_raw_den"], float(expected), places=12)

    def test_rejects_invalid_shapes_and_epsilon(self):
        with self.assertRaises(ValueError):
            paired_band_metrics(np.empty((0, 2, 1)), np.zeros(1))
        with self.assertRaises(ValueError):
            paired_band_metrics(np.zeros((1, 1, 1)), np.zeros(1))
        with self.assertRaises(ValueError):
            paired_band_metrics(np.zeros((1, 2, 1)), np.zeros(1), eps=0.0)
