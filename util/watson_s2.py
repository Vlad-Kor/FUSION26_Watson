from scipy.special import erf, erfi, erfinv
import numpy as np


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


def sample_closed(sample_options, distribution_options):

	sample_count = sample_options[0].state
	k = distribution_options[0].state # kappa

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