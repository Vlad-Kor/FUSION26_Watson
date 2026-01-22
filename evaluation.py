import numpy as np
import sympy as sp
import pandas as pd
from util.watson_s2 import sample_rank1, sample_random, sample_kronecker
import matplotlib.pyplot as plt


x_0 = np.array([4, 5, 6])
x_0 = np.array([4., 5., 6.]) / np.linalg.norm(np.array([4., 5., 6.]))

# samples are shape (N, 3)
methods = {
	"rank1": sample_rank1,
	"kronecker": sample_kronecker,
	"random": sample_random
}

def g(x):
	return np.linalg.norm(x - x_0, axis=-1)

def calc_with_method(samplecount, kappa, method,  idx=None):
	assert (method != "rank1" or idx is not None), f"for {method} method, fib_idx has to be provided"
	res = g(methods[method](kappa, samplecount, idx))
	L = res.shape[0]
	print(f"method: {method}, samplecount: {samplecount}, kappa: {kappa}, result samples: {L}")

	res = 1/L * np.sum(res)
	return res


def get_sample_counts(max_count=1000, max_fib_idx=16):
	#reg = np.linspace(1, max_count+1, 100, dtype=int)
	reg = np.unique( np.logspace(0, np.log10(max_count), 50, dtype=int) )
	fib_idxes = np.arange(4, max_fib_idx)
	fib_numbers = [int(sp.fibonacci(i)) -1 for i in fib_idxes]

	return reg, fib_numbers, fib_idxes


def evalute_all_methods(kappa, counts):
	samplecounts, fib_numbers, fib_idxes = counts
	all_counts = sorted(set(samplecounts).union(fib_numbers))
	data = pd.DataFrame(index=all_counts, columns=methods.keys(), dtype=float)
	data.index.name = "sample_count"

	for method in methods.keys():
		if method == "rank1":
			for fc, fi in zip(fib_numbers, fib_idxes):
				data.loc[fc, method] = calc_with_method(fc, kappa, method, fi)
		else:
			for sc in samplecounts:
				data.loc[sc, method] = calc_with_method(sc, kappa, method)

	return data.sort_index()
		

def get_error(data, kappa):
	#reference_method = "kronecker"
	reference_method = "random"
	samplecount = 1000000

	ref_val = calc_with_method(samplecount, kappa, reference_method)
	return (data - ref_val).abs()
					


		



def plot_data(data, kappa):

	plt.figure()
	for method in data.columns:
		s = data[method].dropna()
		plt.plot(s.index, s, label=method)
	plt.yscale("log")
	plt.xscale("log")
	plt.xlabel("Sample Count")
	plt.ylabel("Error")
	plt.legend()
	plt.show()



if __name__ == "__main__":
	counts = get_sample_counts(max_count=1000, max_fib_idx=16)
	kappa = 10
	data = evalute_all_methods(kappa, counts)
	error_data = get_error(data, kappa)
	plot_data(error_data, kappa)
