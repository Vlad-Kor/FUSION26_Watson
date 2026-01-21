import numpy as np
from util.watson_s2 import sample_rank1, sample_random, sample_kronecker

x_0 = np.array([4, 5, 6])

# samples are shape (N, 3)
methods = {
	"rank1": sample_rank1,
	"kronecker": sample_kronecker,
	"random": sample_random
}

def g(x):
	return np.linalg.norm(x - x_0, axis=-1)

def calc_with_method(samplecount, kappa, method,  idx=None):
	assert (method != "rank1" or idx is not None), "for rank1 method, fib_idx has to be provided"
	L = samplecount

	res = 1/L * np.sum(g(methods[method](kappa, samplecount, idx)))
	return res


if __name__ == "__main__":
	t1 = calc_with_method(21, 10, "rank1", 8)
	t2 = calc_with_method(21, 10, "kronecker")
	t3 = calc_with_method(21, 10, "random")

	truth = calc_with_method(1000000, 10, "kronecker")

	print("rank1:", t1)
	print("kronecker:", t2)
	print("random:", t3)

	print(f"errors: rank1={abs(t1-truth)}, kronecker={abs(t2-truth)}, random={abs(t3-truth)}")