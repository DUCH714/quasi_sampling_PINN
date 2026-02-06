import numpy as np
import jax.numpy as jnp
import jax

def normalization(interval, dim, is_normalization,is_t=0):
    '''
    Creates a normalization function that maps data from a given interval to [-1, 1].
    
    :param interval: list/tuple of [min, max] values defining the normalization range
    :param dim: int, the dimension of the input data (number of features/coordinates)
    :param is_normalization: int, flag to enable/disable normalization
                             - 0: no normalization (identity function)
                             - 1: apply normalization to [-1, 1]
    :param is_t: int, for time dimension handling (0: no time dimension, 1: last dimension is time)
    :return: lambda function that normalizes input data
    '''
    if is_normalization == 1:
        max = interval[1] * jnp.ones(dim-is_t)
        min = interval[0] * jnp.ones(dim-is_t)
        mean = (max + min) / 2
        if is_t==0:
            # Normalize all dimensions to [-1, 1]
            x_fun = lambda x: 2 * (x - mean) / (max - min)
        else:
            # Normalize spatial dimensions, preserve time dimension
            x_fun = lambda x: jnp.stack([2 * (x[:-1] - mean) / (max - min), x[-1]])
    else:
        # No normalization - return identity function
        x_fun = lambda x: x
    return x_fun
