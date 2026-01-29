import unittest

import numpy as np

from sample import _get_fibonacci_index, _is_fibonacci, sample


class TestSample(unittest.TestCase):
	def test_fibonacci_helpers(self):
		self.assertTrue(_is_fibonacci(8))
		self.assertFalse(_is_fibonacci(7))
		self.assertEqual(_get_fibonacci_index(13), 7)  # F_7 = 13

	def test_invalid_sample_count(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=2, sample_count=0, method="random")

	def test_invalid_dim(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=4, sample_count=1, method="random")

	def test_invalid_method(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=2, sample_count=1, method="rank1")

	def test_fibonacci_rank1_requires_plus_one(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=2, sample_count=8, method="fibonacci-rank1")

	def test_s3_limits(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=3, sample_count=178, method="kronecker")
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=3, sample_count=156, method="rank1")

	def test_rank1_cbc_invalid_count(self):
		with self.assertRaises(ValueError):
			sample(kappa=0.0, dim=3, sample_count=4, method="rank1_cbc")

	def test_s2_random_shape(self):
		points = sample(kappa=0.0, dim=2, sample_count=5, method="random")
		self.assertEqual(points.shape, (5, 3))
		norms = np.linalg.norm(points, axis=1)
		self.assertTrue(np.allclose(norms, 1.0, atol=1e-7))


if __name__ == "__main__":
	unittest.main()
