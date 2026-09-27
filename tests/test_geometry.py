import unittest

import numpy as np

from hsrg.feature_maps import BandMixingMap, IdentityMap
from hsrg.geometry import anchored_band_orthogonality_report, band_orthogonality_report, pairwise_isometry_report


class GeometryTests(unittest.TestCase):
    def test_affine_offset_separates_historical_and_anchored_coherence(self):
        components = np.eye(2)[:, None, :]
        affine = lambda x: x.reshape(len(x), -1) + np.array([10.0, 10.0])
        historical = band_orthogonality_report(components, affine)
        anchored = anchored_band_orthogonality_report(components, affine)
        self.assertGreater(historical.output_mean_abs_offdiag_cosine, 0.99)
        self.assertLess(anchored.output_mean_abs_offdiag_cosine, 1e-12)

    def test_anchored_near_zero_responses_are_excluded(self):
        components = np.eye(2)[:, None, :]
        zero = lambda x: np.zeros((len(x), 3))
        report = anchored_band_orthogonality_report(components, zero, eps=1e-12)
        self.assertEqual(report.active_bands, 0)
        self.assertEqual(report.excluded_near_zero_bands, 2)
        self.assertTrue(np.isnan(report.output_mean_abs_offdiag_cosine))

    def test_identity_preserves_pairwise_geometry_after_scaling(self):
        x = np.random.default_rng(3).normal(size=(16, 2, 32))
        report = pairwise_isometry_report(x, IdentityMap(), dt=0.25, n_pairs=128, seed=4)
        self.assertLess(report.scaled_abs_ratio_error_mean, 1e-10)
        self.assertAlmostEqual(report.distance_spearman, 1.0, places=10)

    def test_band_mixer_increases_coherence_of_orthogonal_components(self):
        t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        bands = np.stack([np.sin(t), np.cos(t)])[:, None, :]
        identity = band_orthogonality_report(bands, IdentityMap())
        mixed = band_orthogonality_report(bands, BandMixingMap())
        self.assertLess(identity.output_mean_abs_offdiag_cosine, 1e-10)
        self.assertGreater(mixed.output_mean_abs_offdiag_cosine, 0.1)


if __name__ == "__main__":
    unittest.main()
