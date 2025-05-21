import numpy as np
from numpy import arange, exp, cos, sin, e, pi, absolute, meshgrid
import scipy
from scipy.special import ellipj, ellipkinc, ellipeinc, jn, yn, lpmv, sph_harm, gamma
from scipy.integrate import quad


def get_data(datatype):
    if datatype == 'allen_cahn':
        generate_data = allen_cahn
    elif datatype == 'poisson':
        generate_data = poisson
    elif datatype == 'poisson_sin':
        generate_data = poisson_sin
    elif datatype == 'sine_gordon':
        generate_data = sine_gordon
    else:
        assert False, f'{datatype} does not exist'
    return generate_data


def poisson(x, alpha):
    y = np.exp(-alpha * np.sum(x ** 2, axis=1))[:, None]
    return y


def allen_cahn(x, c):
    A = np.sum(c*np.sin(x[:,:-1]+np.cos(x[:,1:])+x[:,1:]*np.sin(x[:,:-1])),axis=1)
    B = 1 - np.mean(x ** 2, axis=1)
    return (A*B)[:,None]


def sine_gordon(x, alpha, c):
    A = np.mean(np.exp(-c * x[:, :-2] * x[:, 1:-1] * x[:, 2:]), axis=1)
    B = -alpha * np.sum(x ** 2, axis=1)
    return (A * np.exp(B))[:, None]


def poisson_sin(x, dim):
    temp = np.sum(x, axis=1) / dim
    y = (temp) ** 2 + np.sin(temp)
    return y[:, None]
