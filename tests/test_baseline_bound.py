import unittest

import numpy as np

from hsrg.audit_metrics import paired_band_metrics


class BaselineBoundTest(unittest.TestCase):
    """Numerical checks of Proposition 1 and Remark 1 of the article."""

    def test_decomposition_and_raw_bound(self):
        rng = np.random.default_rng(0)
        for eps in (0.05, 0.1, 0.2, 0.3):
            baseline = rng.normal(size=64)
            residual = rng.normal(size=(1, 5, 64))
            residual *= eps * np.linalg.norm(baseline) / np.linalg.norm(residual, axis=-1, keepdims=True)
            row = paired_band_metrics(baseline + residual, baseline)[0]
            signed = row["baseline_sq_over_raw_den"] + row["cross_over_raw_den"] + row["residual_over_raw_den"]
            self.assertAlmostEqual(signed, row["raw_signed_mean"], places=12)
            bound = (1 - 2 * eps - eps ** 2) / (1 + eps) ** 2
            self.assertGreaterEqual(row["raw_odi"], bound - 1e-12)

    def test_orthogonal_components_with_offset(self):
        baseline = np.array([10.0, 10.0, 0.0])
        responses = baseline + np.eye(3)[None, :2, :]
        row = paired_band_metrics(responses, baseline)[0]
        self.assertAlmostEqual(row["anchored_odi"], 0.0, places=12)
        self.assertAlmostEqual(row["raw_odi"], 220.0 / 221.0, places=12)

    def test_anchor_far_from_responses(self):
        rng = np.random.default_rng(1)
        baseline = rng.normal(size=32) * 100.0
        responses = rng.normal(size=(1, 5, 32))
        eps = float(np.linalg.norm(responses, axis=-1).max() / np.linalg.norm(baseline))
        row = paired_band_metrics(responses, baseline)[0]
        self.assertGreaterEqual(row["anchored_odi"], 1 - 4 * eps)


if __name__ == "__main__":
    unittest.main()
