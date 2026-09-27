import math
import unittest

import numpy as np
from scipy.special import gammaln

from scripts.reaggregate_analytic_gaussian_null import analytic_gaussian_alpha


def alpha_d(dimension: int) -> float:
    return math.exp(gammaln(dimension / 2) - 0.5 * math.log(math.pi) - gammaln((dimension + 1) / 2))


class AnalyticGaussianTests(unittest.TestCase):
    def test_closed_form_matches_monte_carlo(self):
        rng = np.random.default_rng(7)
        for dimension in (2, 8, 32):
            left = rng.normal(size=(100_000, dimension))
            right = rng.normal(size=(100_000, dimension))
            estimate = np.mean(np.abs(np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))))
            self.assertAlmostEqual(estimate, alpha_d(dimension), delta=0.004)

    def test_release_formula_matches_reference_formula(self):
        for dimension in (1, 2, 16, 256):
            self.assertAlmostEqual(analytic_gaussian_alpha(dimension), alpha_d(dimension), delta=1e-13)
