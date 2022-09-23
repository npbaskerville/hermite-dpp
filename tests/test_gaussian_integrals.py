import unittest

import numpy as np
from parameterized import parameterized
from scipy.special import factorial2
from scipy import stats
from hermitedpp import MultivariateHermiteFactorOPE


def gaussian_moment(p):
    """
    Compute the p-th moment of univariate standard Gaussian.
    Vectorised over p.

    :param float,array-like p: The moment(s) to compute.
    :rtype: np.ndarray
    :returns: The corresponding Gaussian moments.
     """
    _p = np.asarray(p)
    return factorial2(_p-1) * (1 - _p % 2) * np.sqrt(2 * np.pi)


def EZ_estimator(integrand, dpp, sample=None):
    """Estimate integrand using naive EZ integration against Gaussian measure."""
    if sample is not None:
        phi_x = dpp.eval_multi_dimensional_polynomials(sample)
        integrand_x = integrand(sample).ravel()

        estimate = np.linalg.solve(phi_x.astype(np.float64), integrand_x.astype(np.float64))[0]
        estimate *= np.sqrt(dpp.mass_of_mu)
        
        return estimate
    else:
        sample = dpp.sample()
        return EZ_estimator(integrand, dpp, sample)


class GaussianMomentTests1D(unittest.TestCase):
    """Test EZ integration of Gaussian moments against known values in 1d."""
    MOMENTS = [2, 3, 6]
    N = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpp = MultivariateHermiteFactorOPE(self.N, 1)
        self.sample = self.dpp.sample()

    def _moment_function(self, ind):
        def moment(x):
            return np.prod(x**ind, axis=-1)
        return moment

    @parameterized.expand([(i, ) for i in range(3)])
    def test_ez_estimator(self, moment_ind):
        integrand = self._moment_function(self.MOMENTS[moment_ind])
        estimate = EZ_estimator(integrand, self.dpp, self.sample)
        true_value = np.prod(gaussian_moment(self.MOMENTS[moment_ind]))
        self.assertAlmostEqual(estimate, true_value, places=4)

class GaussianMomentTests2D(unittest.TestCase):
    """Test EZ integration of Gaussian moments against known values in 2d."""
    MOMENTS = [(2, 1), (2, 3), (1, 6)]
    N = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpp = MultivariateHermiteFactorOPE(self.N, 2)
        self.sample = self.dpp.sample()

    def _moment_function(self, ind):
        def moment(x):
            return np.prod(x**ind, axis=-1)
        return moment

    @parameterized.expand([(i, ) for i in range(3)])
    def test_ez_estimator(self, moment_ind):
        integrand = self._moment_function(self.MOMENTS[moment_ind])
        estimate = EZ_estimator(integrand, self.dpp, self.sample)
        true_value = np.prod(gaussian_moment(self.MOMENTS[moment_ind]))
        self.assertAlmostEqual(estimate, true_value, places=4)
