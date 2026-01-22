import numpy as np
import sympy as sp
import pandas as pd
from util.watson_s2 import sample_rank1, sample_random, sample_kronecker
from util.watson_s3 import sample_frolov_s3, sample_sobol_s3, sample_random_s3
import matplotlib.pyplot as plt

EVAL_TIMES_FOR_RANDOM = 100
REFERENCE_METHOD = "random"
REFERENCE_SAMPLECOUNT = 1000000
x_0 = np.array([4, 5, 6])
x_0_s3 = np.array([4, 5, 6, 7])

# samples are shape (N, 3)
methods_s2 = {
	"rank1": sample_rank1,
	"kronecker": sample_kronecker,
	"random": sample_random
}

methods_s3 = {
	"frolov_s3": sample_frolov_s3,
	"sobol_s3": sample_sobol_s3,
	"random": sample_random_s3
}

def g(x):
	if x.shape[1] == 4:
		_x_0 = x_0_s3
	else:
		_x_0 = x_0
	return np.linalg.norm(x - _x_0, axis=-1)

def calc_with_method(samplecount, kappa, method,  idx=None, _methods=None):
	assert (method != "rank1" or idx is not None), f"for {method} method, fib_idx has to be provided"
	res = g(_methods[method](kappa, samplecount, idx))
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


def evalute_all_methods(kappa, counts, ref_val=None, _methods=None):


	samplecounts, fib_numbers, fib_idxes = counts
	all_counts = sorted(set(samplecounts).union(fib_numbers))
	data = pd.DataFrame(index=all_counts, columns=_methods.keys(), dtype=float)
	data.index.name = "sample_count"

	for method in _methods.keys():
		if method == "rank1":
			for fc, fi in zip(fib_numbers, fib_idxes):
				val = calc_with_method(fc, kappa, method, fi, _methods=_methods)
				if ref_val is not None:
					val = np.abs(val - ref_val)
				data.loc[fc, method] = val
		elif method == "random":
			for sc in samplecounts:
				vals = np.array([
					calc_with_method(sc, kappa, method, _methods=_methods)
					for _ in range(EVAL_TIMES_FOR_RANDOM)
				])
				if ref_val is not None:
					errs = np.abs(vals - ref_val)
					data.loc[sc, method] = np.mean(errs)
					data.loc[sc, method+"_std"] = np.std(errs)
				else:
					data.loc[sc, method] = np.mean(vals)
					data.loc[sc, method+"_std"] = np.std(vals)
		else:
			for sc in samplecounts:
				val = calc_with_method(sc, kappa, method, _methods=_methods)
				if ref_val is not None:
					val = np.abs(val - ref_val)
				data.loc[sc, method] = val

	return data.sort_index()
		

def get_reference_value(kappa, _methods):
	return calc_with_method(REFERENCE_SAMPLECOUNT, kappa, REFERENCE_METHOD, _methods=_methods)


def get_error(data, kappa, ref_val=None, _methods=None):
	if ref_val is None:
		ref_val = get_reference_value(kappa, _methods=_methods)
	error = data.copy()
	for col in error.columns:
		if col.endswith("_std"):
			continue
		error[col] = (error[col] - ref_val).abs()
	return error
					


		



def plot_data(data, kappa):
	plt.close("all")
	plt.figure()
	for method in data.columns:
		if method.endswith("_std"):
			continue

		s = data[method].dropna()
		if s.empty:
			continue
		(line,) = plt.plot(s.index, s, label=method)

		std_col = f"{method}_std"
		if std_col in data.columns:
			std = data[std_col].reindex(s.index)
			band = std.notna()
			if band.any():
				upper = s[band] + std[band]
				lower = s[band] - std[band]
				color = line.get_color()
				valid = lower > 0
				upper = upper.where(valid)
				lower = lower.where(valid)
				plt.plot(upper.index, upper, color=color, linewidth=0.6, alpha=0.7)
				plt.plot(lower.index, lower, color=color, linewidth=0.6, alpha=0.7)
				plt.fill_between(upper.index, lower, upper, color=color, alpha=0.15)
	plt.yscale("log")
	plt.xscale("log")
	plt.xlabel("Sample Count")
	plt.ylabel("Error")
	plt.legend()
	plt.show()



if __name__ == "__main__":
	# evaluate S2 methods
	counts = get_sample_counts(max_count=1000, max_fib_idx=16)
	kappa = 10
	ref_val = get_reference_value(kappa, _methods=methods_s2)
	error_data = evalute_all_methods(kappa, counts, ref_val=ref_val, _methods=methods_s2)
	plot_data(error_data, kappa)

	# evaluate S3 methods
	counts = get_sample_counts(max_count=1000, max_fib_idx=16)
	kappa = 10
	ref_val = get_reference_value(kappa, _methods=methods_s3)
	error_data = evalute_all_methods(kappa, counts, ref_val=ref_val, _methods=methods_s3)
	plot_data(error_data, kappa)
