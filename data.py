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
    '''
    Poisson equation: y = exp(-alpha * ||x||^2)
    '''
    y = np.exp(-alpha * np.sum(x ** 2, axis=1))[:, None]
    return y


def allen_cahn(x, c):
    '''
    Allen-Cahn equation: y = (sum_i c*sin(x_i + cos(x_{i+1}) + x_{i+1}*sin(x_i))) * (1 - mean(x^2))
    '''
    A = np.sum(c*np.sin(x[:,:-1]+np.cos(x[:,1:])+x[:,1:]*np.sin(x[:,:-1])),axis=1)
    B = 1 - np.mean(x ** 2, axis=1)
    return (A*B)[:,None]


def sine_gordon(x, c):
    '''
    Sine-Gordon equation: y = (sum_i (exp(-c * x_i * x_{i+1} * x_{i+2}))/dim) * (1 - sum_i x_i^2/dim)
    '''
    A = np.mean(np.exp(-c * x[:, :-2] * x[:, 1:-1] * x[:, 2:]), axis=1)
    B = 1 - np.mean(x ** 2, axis=1)
    return (A * B)[:, None]


def poisson_sin(x, dim):
    '''
    Poisson-Sine equation: y = (sum__i x_i / dim)^2 + sin(sum__i x_i / dim)
    '''
    s = np.sum(x, axis=1) / dim
    y = (s) ** 2 + np.sin(s)
    return y[:, None]
