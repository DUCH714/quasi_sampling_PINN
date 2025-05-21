import numpy as np
import skopt

def get_sampler(sampling_mode):
    if sampling_mode == 'halton':
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampling_mode == 'sobol':
        sampler = skopt.sampler.Sobol(skip=0, randomize=False)
    else:
        assert False, f'{sampling_mode} does not exist'
    return sampler

# n_samples=100
# space = [(-1.0, 1.0), (-1.0, 1.0)]
# sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
# points=np.array(sampler.generate(space, n_samples))
# print(points)
#
# sampler = skopt.sampler.Sobol(skip=0, randomize=False)