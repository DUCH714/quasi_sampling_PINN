from jax import random
import jax.numpy as jnp
import numpy as np

def scoring_function(eigvects, eigvals, K_test,residual,eps_ntk=1e-8):
    '''
    Using Nystrom approximation to get alpha
    :param eigvects: v in (14)
    :param eigvals: \lambda in (14)
    :param K_train_test: \Theta in (14)
    :param residual: R_{theta} in (14)
    :param eps_ntk:
    :return: score for each point in the pool
    '''
    idxs = eigvals > eps_ntk
    eigvals_chopped = eigvals[idxs].reshape(-1,1)
    eigvects_chopped = eigvects[:, idxs]
    P = (1. / (eigvals_chopped ** 0.5)) * (eigvects_chopped.T @ K_test) * residual.reshape(1,-1)
    return P

def do_sampling(K_train,K_test,Z_pool,Z_pool_res,num_points,key,eps_ntk=1e-8):
    '''
    Performs active sampling based on NTK (Neural Tangent Kernel) scoring.
    Selects points from the pool with the highest information content using Nystrom approximation.
    
    :param K_train: NTK matrix of training points, shape (n_train, n_train)
    :param K_test: NTK matrix between training and candidate points, shape (n_train, n_pool)
    :param Z_pool: Candidate points for sampling, shape (n_pool, d)
    :param Z_pool_res: Residuals at candidate points, shape (n_pool,)
    :param num_points: Number of points to sample from the pool
    :param key: JAX random key
    :param eps_ntk: Small constant for numerical stability (default: 1e-8)
    :return: Selected points from Z_pool, shape (num_points, d)
    '''
    eigvals, eigvects = jnp.linalg.eigh(K_train + eps_ntk * jnp.eye(K_train.shape[0]))
    alpha = scoring_function(eigvects, eigvals, K_test, Z_pool_res)
    probs = np.array(jnp.linalg.norm(alpha, axis=0) ** 2) + 1e-9
    # Sample points according to computed probabilities
    x_points=random.choice(key, Z_pool, shape=(num_points,),p=probs, replace=False)
    return x_points
