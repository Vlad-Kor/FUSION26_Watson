from functools import lru_cache
import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import sphstat
from pyrecest.backend import array
from pyrecest.distributions import WatsonDistribution as WatsonDistributionPyrecest
from scipy.special import erf, erfi, erfinv
from scipy.optimize import brentq
from deterministic_gaussian_sampling_fibonacci import get_uniform_grid
import matplotlib.pyplot as plt
from util.generators import rank1, rank1_cbc


def spherical_to_cartesian(psi, theta, phi):
	x1 = np.cos(psi)
	x2 = np.sin(psi) * np.cos(theta)
	x3 = np.sin(psi) * np.sin(theta) * np.cos(phi)
	x4 = np.sin(psi) * np.sin(theta) * np.sin(phi)
	return np.column_stack((x1, x2, x3, x4))

@lru_cache
def get_inverse_interpolated(kappa): 
	mu = array([1.0, 0.0, 0.0, 0.0])
	watson_dist = WatsonDistributionPyrecest(mu=mu, kappa=kappa)
	

	
	def pdf(psi, theta, phi):
		x = spherical_to_cartesian(psi=psi, theta=theta, phi=phi)
		wts = watson_dist.pdf(array(x))
		return wts.item()

	def f(t,y):
		p = float(pdf(t, 0.0, 0.0))  # scalar
		return [4.0 * np.pi * p * (np.sin(t) ** 2)]
	
	t_span = (0, np.pi) # psi from 0 to pi
	y0 = 0 # the value of the integrated pdf at 0 is 0

	sol = scipy.integrate.solve_ivp(f, t_span, [y0], rtol=1e-9, atol=1e-12)

	x = sol.t
	y = sol.y[0]

	
	# due to numerical issues, for large kappa and samplecount, y can be slightly non monotonic
	# monotonicity is needed for interpolation, so maximum.accumulate then bump by eps
	y = np.maximum.accumulate(y) 
	y /= y[-1] # normalize to [0,1]
	eps = 1e-14
	y += eps * np.arange(len(y)) 

	# now interpolate, but we swamp x and y so whe get the inverse function
	# this works because the function is monotonic
	# use PCHIP interpolation
	q1 = scipy.interpolate.PchipInterpolator(x=y, y=x)
	return q1

def sample_inverse_interpolation(grid, kappa):
	q1 = get_inverse_interpolated(kappa)	
	
	def q2(u):
		# inverse of -(1/2) * (cos(x)-1)
		return np.arccos(1.0 - 2.0*u)
	
	
	def q3(phi):
		return 2 * np.pi * phi
	

	grid_psi = q1(grid[:,0])
	grid_theta = q2(grid[:,1])
	grid_phi = q3(grid[:,2])

	samples = spherical_to_cartesian(psi=grid_psi, theta=grid_theta, phi=grid_phi)
	return samples


def grid_random(sample_count):
	grid = np.random.uniform(0.0, 1.0, size=(sample_count, 3))
	return grid

def grid_sobol(sample_count):
	from scipy.stats import qmc
	sobol_engine = qmc.Sobol(d=3, scramble=True)
	sobol_samples = sobol_engine.random(n=sample_count)
	return sobol_samples

def grid_frolov(sample_count):
	grid = get_uniform_grid(3, sample_count, "Fibonacci")
	return grid

def grid_kronecker(sample_count):
	assert sample_count <= 177, "Kronecker grid only implemented up to 177 samples"

	# see https://publikationen.bibliothek.kit.edu/1000179985 5.3.2.B; C.2.2: L<=177
	g0 = 1/sample_count
	g1 = 0.38196935570538115
	g2 = 0.42019917392673339 
	i = np.arange(0, sample_count)


	psi = (i*g0) % 1
	theta = (i*g1) % 1
	phi = (i*g2) % 1

	grid = np.column_stack((psi, theta, phi))
	return grid


def latice_from_generator(sample_count, generator):
	indices = np.arange(1, sample_count+1)
	psi = ((indices/sample_count) * generator[0]) % 1
	theta = ((indices/sample_count) * generator[1]) % 1
	phi = ((indices/sample_count) * generator[2]) % 1

	grid = np.column_stack((psi, theta, phi))
	return grid

def grid_rank1(sample_count):
	generator = rank1[sample_count]
	generator = generator[1:]
	print(generator)
	grid = latice_from_generator(sample_count, generator)
	return grid

def grid_rank1_cbc(sample_count):
	generator = rank1_cbc[sample_count]
	grid = latice_from_generator(sample_count, generator)
	return grid


	
def sample_random_s3(kappa, sample_count, _):
	grid = grid_random(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples

def sample_sobol_s3(kappa, sample_count, _):
	grid = grid_sobol(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples

def sample_frolov_s3(kappa, sample_count, _):
	grid = grid_frolov(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples

def sample_kronecker_s3(kappa, sample_count, _):
	grid = grid_kronecker(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples

def sample_rank1_s3(kappa, sample_count, _):
	grid = grid_rank1(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples

def sample_rank1_cbc_s3(kappa, sample_count, _):
	grid = grid_rank1_cbc(sample_count)
	samples = sample_inverse_interpolation(grid, kappa)
	return samples
