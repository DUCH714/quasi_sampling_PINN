import numpy as np
import skopt

def get_sampler(sampling_mode):
    '''
    Returns a quasi-random sampler based on the specified sampling mode.
    Quasi-random sampling methods are used to generate 
    sample points across this sampler.
    
    :param sampling_mode: str, the type of quasi-random sampler to use
                          - 'halton': Halton low-discrepancy sequence
                          - 'sobol': Sobol low-discrepancy sequence
    :return: sampler object that can generate quasi-random samples
    '''
    if sampling_mode == 'halton':
        # Halton sequence
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampling_mode == 'sobol':
        # Sobol sequence
        sampler = skopt.sampler.Sobol(skip=0, randomize=False)
    else:
        assert False, f'{sampling_mode} does not exist'
    return sampler