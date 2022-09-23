import unittest

from hermitedpp.multivariate_hermite_factor_ope import compute_ordering
 

class TestMultiIndexOrdering(unittest.TestCase):
    def test_2d(self):
        base_order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (0, 3), (1, 3), (2, 3)]
        self.assertEqual(compute_ordering(12, 2), base_order)
        self.assertEqual(compute_ordering(11, 2), base_order[:11])

    def test_3d(self):
        base_order = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (0, 0, 2)]
        self.assertEqual(compute_ordering(9, 3), base_order)