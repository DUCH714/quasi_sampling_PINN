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

class interior_points():
    def __init__(self, dim, interval=(-1, 1)):
        self.dim = dim
        self.interval = interval

    def sample(self, num, key):
        points = random.uniform(key,shape=(num,self.dim),minval=self.interval[0],maxval=self.interval[1])
        return points

class boundary_points():
    def __init__(self, dim, generate_data, interval=(-1, 1)):
        self.dim = dim
        self.points = jnp.linspace(interval[0], interval[1], 100)
        self.interval = interval
        self.generate_data = generate_data

    def sample(self, num, key):
        keys = random.split(key, self.dim + 1)
        x = jnp.concatenate([random.choice(key, self.points, shape=(num, 1), replace=True) for key in keys[:-1]], -1)
        keys = random.split(keys[-1], 2)
        boundary = jax.random.randint(keys[0], (num,), 0, 2) * (self.interval[1] - self.interval[0]) + self.interval[0]
        idx_bd = jax.random.randint(keys[1], (num,), 0, self.dim)
        vset = lambda p, idx, value: p.at[idx].set(value)
        x = vmap(vset, (0, 0, 0))(x, idx_bd, boundary)
        y = self.generate_data(x)
        return x, y