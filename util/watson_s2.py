from scipy.special import erf, erfi, erfinv
from scipy.stats import qmc
import numpy as np
import sphstat
import sympy as sp


def erfi_inv(y, iters=8, thresh=1.5):
	y = np.asarray(y, dtype=float)
	sgn = np.sign(y)
	z = np.abs(y)

	x = np.zeros_like(z)
	mask = z > 0
	if not np.any(mask):
		return x  # all zeros

	z_nz = z[mask]
	x0 = np.empty_like(z_nz)

	# Small region: Taylor
	small = z_nz <= thresh
	large = ~small

	x0[small] = z_nz[small] * np.sqrt(np.pi) / 2.0

	# Large region: leading asymptotic, ignoring 1/x
	if np.any(large):
		t = np.log(z_nz[large] * np.sqrt(np.pi))
		t = np.maximum(t, 0.0)
		x0[large] = np.sqrt(t)

	# Newton
	x_new = x0
	for _ in range(iters):
		fx = erfi(x_new) - z_nz
		dfx = 2.0 / np.sqrt(np.pi) * np.exp(x_new**2)
		x_new = x_new - fx / dfx

	x[mask] = x_new
	return sgn * x


def sample_kronecker(k, sample_count, _):
	gold_seq = (1+5**0.5)/2  # golden ratio

	indices = np.arange(0, sample_count)
	
	if k > 0:
		w = 1 / (np.sqrt(k)) * erfi_inv( ((1-2*indices + sample_count)/ sample_count) * erfi(np.sqrt(k)) )
	elif k < 0:
		la = -k
		w = 1 / (np.sqrt(la)) * erfinv( ((2*indices +1 - sample_count)/ sample_count) * erf(np.sqrt(la)) )
	elif k == 0:
		w = ((2*indices +1 - sample_count)/ sample_count)


	w = np.clip(w, -1.0, 1.0) # clamp to avoid sqrt warnings due to numerical issues

	x_i_f_0 = w
	x_i_f_1 = np.sqrt(1-w**2) * np.cos( (2 * np.pi * indices) / gold_seq)
	x_i_f_2 = np.sqrt(1-w**2) * np.sin( (2 * np.pi * indices) / gold_seq)
	x_i_f = np.column_stack((x_i_f_0, x_i_f_1, x_i_f_2))

	return x_i_f

def sample_random(kappa, sample_count, _):
	k = kappa
	xy = np.random.uniform(0, 1, size=(2, sample_count))
	x, y = xy[0], xy[1]

	x = 2*x -1  # map x from [0,1] to [-1, 1]
	phi = 2 * np.pi * y  # azimuthal angle, [0, 2pi] uniform

	if k > 0:
		w = 1 / (np.sqrt(k)) * erfi_inv( x * erfi(np.sqrt(k)) )
	elif k < 0:
		la = -k
		w = 1 / (np.sqrt(la)) * erfinv( x * erf(np.sqrt(la)) )
	elif k == 0:
		w = x


	w = np.clip(w, -1.0, 1.0) # clamp to avoid sqrt warnings due to numerical issues

	x_i_f_0 = w
	x_i_f_1 = np.sqrt(1-w**2) * np.cos( phi)
	x_i_f_2 = np.sqrt(1-w**2) * np.sin( phi)
	x_i_f = np.column_stack((x_i_f_0, x_i_f_1, x_i_f_2)) # order so that mu=[1, 0, 0]
	return x_i_f


def get_rank_1(sample_count, k, without_first_point=False):
	indices = np.arange(0, sample_count+1)

	
	# centered rank-1 lattice
	F_k = int(sp.fibonacci(k - 1))
	F_k_p_1 = sample_count+1  # int(sp.fibonacci(k))
	assert F_k_p_1 == int(sp.fibonacci(k)), "sample_count has to be the k-th fibonacci number "
	
	z = (indices * (1/F_k_p_1) + (1/(2*F_k_p_1)) ) % 1
	p = (indices * (F_k/F_k_p_1) + (1/(2*F_k_p_1)) ) % 1

	if without_first_point:
		z = z[1:]
		p = p[1:]

	return z, p

"""
Note that sample_count has to be a fibonacci number
"""
def sample_rank1(k, sample_count, fib_idx):


		x, y = get_rank_1(sample_count, fib_idx, without_first_point=True)

		x = 2*x -1  # map x from [0,1] to [-1, 1]
		phi = 2 * np.pi * y  # azimuthal angle, [0, 2pi] uniform

		if k > 0:
			w = 1 / (np.sqrt(k)) * erfi_inv( x * erfi(np.sqrt(k)) )
		elif k < 0:
			la = -k
			w = 1 / (np.sqrt(la)) * erfinv( x * erf(np.sqrt(la)) )
		elif k == 0:
			w = x


		w = np.clip(w, -1.0, 1.0) # clamp to avoid sqrt warnings due to numerical issues

		x_i_f_0 = w
		x_i_f_1 = np.sqrt(1-w**2) * np.cos( phi)
		x_i_f_2 = np.sqrt(1-w**2) * np.sin( phi)
		x_i_f = np.column_stack((x_i_f_0, x_i_f_1, x_i_f_2)) # order so that mu=[1, 0, 0]
		return x_i_f

def sample_sobol(k, sample_count, _):

	sobol_engine = qmc.Sobol(d=2, scramble=True)
	sobol_samples = sobol_engine.random(n=sample_count)
	x, y = sobol_samples[:,0], sobol_samples[:,1]

	x = 2*x -1  # map x from [0,1] to [-1, 1]
	phi = 2 * np.pi * y  # azimuthal angle, [0, 2pi] uniform

	if k > 0:
		w = 1 / (np.sqrt(k)) * erfi_inv( x * erfi(np.sqrt(k)) )
	elif k < 0:
		la = -k
		w = 1 / (np.sqrt(la)) * erfinv( x * erf(np.sqrt(la)) )
	elif k == 0:
		w = x


	w = np.clip(w, -1.0, 1.0) # clamp to avoid sqrt warnings due to numerical issues

	x_i_f_0 = w
	x_i_f_1 = np.sqrt(1-w**2) * np.cos( phi)
	x_i_f_2 = np.sqrt(1-w**2) * np.sin( phi)
	x_i_f = np.column_stack((x_i_f_0, x_i_f_1, x_i_f_2)) # order so that mu=[1, 0, 0]
	return x_i_f

def spherical_to_cartesian_s2(theta, phi, r=1):
		x = r * np.sin(theta) * np.cos(phi)
		y = r * np.sin(theta) * np.sin(phi)
		z = r * np.cos(theta)

		return x, y, z