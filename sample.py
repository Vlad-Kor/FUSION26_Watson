import numpy as np
import sympy as sp

from util.watson_s2 import sample_kronecker, sample_random, sample_rank1, sample_sobol
from util.watson_s3 import (
	sample_kronecker_s3,
	sample_random_s3,
	sample_rank1_cbc_s3,
	sample_rank1_s3,
	sample_sobol_s3,
)

_S2_METHODS = {"kronecker", "random", "sobol", "fibonacci-rank1"}
_S3_METHODS = {"kronecker", "random", "sobol", "rank1", "rank1_cbc"}


def sample(kappa, dim, sample_count, method):
	if sample_count <= 0:
		raise ValueError("sample_count must be positive")

	if dim == 2:
		if method not in _S2_METHODS:
			raise ValueError(f"unknown S2 method: {method}. Supported: {sorted(_S2_METHODS)}")
		if method == "kronecker":
			return sample_kronecker(kappa, sample_count, None)
		if method == "random":
			return sample_random(kappa, sample_count, None)
		if method == "sobol":
			return sample_sobol(kappa, sample_count, None)
		if method == "fibonacci-rank1":
			# get_rank_1 expects sample_count + 1 to be a Fibonacci number.
			if not _is_fibonacci(sample_count + 1):
				raise ValueError(
					"sample_count + 1 must be a Fibonacci number for rank1 sampling on S2"
				)
			return sample_rank1(kappa, sample_count, _get_fibonacci_index(sample_count + 1))

	if dim == 3:
		if method not in _S3_METHODS:
			raise ValueError(f"unknown S3 method: {method}. Supported: {sorted(_S3_METHODS)}")
		if method == "kronecker":
			if sample_count > 177:
				raise ValueError("sample_count > 177 not supported for kronecker on S3")
			return sample_kronecker_s3(kappa, sample_count, None)
		if method == "random":
			return sample_random_s3(kappa, sample_count, None)
		if method == "sobol":
			return sample_sobol_s3(kappa, sample_count, None)
		if method == "rank1":
			if sample_count > 155:
				raise ValueError("sample_count > 155 not supported for rank1 on S3")
			return sample_rank1_s3(kappa, sample_count, None)
		if method == "rank1_cbc":
			try:
				return sample_rank1_cbc_s3(kappa, sample_count, None)
			except KeyError as exc:
				raise ValueError(
					"sample_count not supported for rank1_cbc on S3 , only prime sample counts are supported"
				) from exc

	raise ValueError(f"dim must be 2 or 3 (got {dim})")


def _is_fibonacci(n):
	if n < 0:
		return False
	# A number is a Fibonacci number if and only if one or both of (5*n^2 + 4)
	# or (5*n^2 - 4) is a perfect square.
	# https://en.wikipedia.org/wiki/Fibonacci_sequence
	test1 = 5 * n * n + 4
	test2 = 5 * n * n - 4

	def is_perfect_square(x):
		s = int(np.sqrt(x))
		return s * s == x

	return is_perfect_square(test1) or is_perfect_square(test2)


def _get_fibonacci_index(x):
	for i in range(0, x + 3):
		if int(sp.fibonacci(i)) == x:
			return i
	raise ValueError(f"{x} is not a Fibonacci number")
