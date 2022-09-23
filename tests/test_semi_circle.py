import unittest

import numpy as np
from parameterized import parameterized

from semi_circle_cython_backend import (sample_semicircle, mixture_probability, 
                                        sample_semi_circle_student_mixture, sample_finite_n_semi_circle)


class SemiCircleTests(unittest.TestCase):
    """Basic tests of semicircle moments and extremes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = np.array([sample_semicircle() for _ in range(1000)])

    def test_semicircle_sample_mean(self):
        self.assertAlmostEqual(np.mean(self.samples), 0., 1)

    def test_semicircle_sample_second_moment(self):
         self.assertAlmostEqual(np.mean(self.samples**2), 1., 1)

    def test_semicircle_sample_third_moment(self):
         self.assertAlmostEqual(np.mean(self.samples**3), 0., 1)

    def test_semicircle_max(self):
        self.assertLessEqual(max(self.samples), 2.1)

    def test_semicircle_min(self):
        self.assertGreaterEqual(min(self.samples), -2.1)


class MixtureTests(unittest.TestCase):
    """Basic tests for semicircle student-t mixture."""
    def test_mixture_probability_valid(self):
        ns = np.arange(2, 200)
        probs = np.asarray([mixture_probability(n) for n in ns])
        np.testing.assert_array_less(probs, 1.)
        np.testing.assert_array_less(-probs, 0.)

    def test_sample_mean(self):
        samples = [sample_semi_circle_student_mixture(0.1) for _ in range(1000)]
        self.assertAlmostEqual(np.mean(samples), 0., 1)


class FiniteSemiCircle(unittest.TestCase):
    """Simple moment and extreme tests for the finite n GUE density"""
    @parameterized.expand([(i, ) for i in [3, 5, 10]])
    def test_sample_mean(self, n):
        samples = sample_finite_n_semi_circle(n, 1000)
        self.assertAlmostEqual(np.mean(samples), 0., 1)

    @parameterized.expand([(i, ) for i in [3, 5, 10]])
    def test_sample_max(self, n):
        samples = sample_finite_n_semi_circle(n, 1000)
        self.assertLessEqual(max(samples), np.sqrt(n)*3 , 1)

    @parameterized.expand([(i, ) for i in [3, 5, 10]])
    def test_sample_min(self, n):
        samples = sample_finite_n_semi_circle(n, 1000)
        self.assertGreaterEqual(min(samples), -np.sqrt(n)*3 , 1)