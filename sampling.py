import numpy as np
import skopt
n_samples=100
space = [(-1.0, 1.0), (-1.0, 1.0)]
sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
points=np.array(sampler.generate(space, n_samples))
print(points)

sampler = skopt.sampler.Sobol(skip=0, randomize=False)