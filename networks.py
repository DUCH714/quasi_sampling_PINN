import equinox as eqx
import jax
from jax import random
from jax.nn import gelu, silu, tanh
import jax.numpy as jnp
import numpy as np


def get_network(args, input_dim, output_dim, interval, normalizer, keys):
    if args.network == 'mlp':
        model = MLP(input_dim=input_dim, output_dim=output_dim, N_features=args.features, N_layers=args.layers,
                    normalizer=normalizer, activation=args.activation,
                    key=keys[0])
    elif args.network == 'modifiedmlp':
        model = modifiedMLP(input_dim=input_dim, output_dim=output_dim, N_features=args.features, N_layers=args.layers,
                            normalizer=normalizer, activation=args.activation,
                            key=keys[0])
    else:
        assert False, f'{args.network} does not exist'
    return model


class MLP(eqx.Module):
    matrices: list
    biases: list
    activation: jax.nn
    normalizer: list

    def __init__(self, input_dim, output_dim, N_features, N_layers, normalizer, key, activation='tanh'):
        keys = random.split(key, N_layers + 1)
        features = [input_dim, ] + [N_features, ] * (N_layers - 1) + [output_dim, ]
        self.matrices = [random.normal(key, (f_in, f_out)) / jnp.sqrt((f_in + f_out) / 2) for f_in, f_out, key in
                         zip(features[:-1], features[1:], keys)]
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out in zip(features[:-1], features[1:])]
        if activation == 'silu':
            self.activation = silu
        elif activation == 'gelu':
            self.activation = gelu
        else:
            self.activation = tanh
        self.normalizer = [normalizer]

    def __call__(self, input, frozen_para):

        f = input @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            # f = tanh(f)
            f = self.activation(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f

    def get_frozen_para(self):
        frozen = []
        return frozen


class modifiedMLP(eqx.Module):
    matrices: list
    biases: list
    matrices_modified: list
    biases_modified: list
    activation: jax.nn
    normalizer: list

    def __init__(self, input_dim, output_dim, N_features, N_layers, normalizer, key, activation='tanh'):
        keys = random.split(key, N_layers + 1)
        features = [input_dim, ] + [N_features, ] * (N_layers - 1) + [output_dim, ]
        self.matrices = [random.normal(key, (f_in, f_out)) / jnp.sqrt((f_in + f_out) / 2) for f_in, f_out, key in
                         zip(features[:-1], features[1:], keys)]
        self.matrices_modified = [
            random.normal(key, (features[0], features[1])) / jnp.sqrt((features[0] + features[1]) / 2),
            random.normal(key, (features[0], features[1])) / jnp.sqrt((features[0] + features[1]) / 2)]

        self.biases = [jnp.zeros((f_out,)) for f_in, f_out in zip(features[:-1], features[1:])]
        self.biases_modified = [jnp.zeros((features[1],)), jnp.zeros((features[1],))]

        if activation == 'silu':
            self.activation = silu
        elif activation == 'gelu':
            self.activation = gelu
        else:
            self.activation = tanh

        self.normalizer = [normalizer]

    def __call__(self, input, frozen_para):
        input = self.normalizer[0](input)
        u = input @ self.matrices_modified[0] + self.biases_modified[0]
        v = input @ self.matrices_modified[1] + self.biases_modified[1]

        u = self.activation(u)
        v = self.activation(v)

        f = input @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            # f = tanh(f)
            f = self.activation(f)
            f = f * u + (1 - f) * v
            f = f @ self.matrices[i] + self.biases[i]
        return f

    def get_frozen_para(self):
        frozen = []
        return frozen
