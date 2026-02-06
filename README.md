# Quasi-random physics-informed neural networks

This repository contains the  code for [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231226003103): Quasi-random physics-informed neural networks

## Overview
This project extends standard PINNs by integrating quasi-random sampling (e.g., Sobol sequences) to replace random sampling points, reducing sampling bias and accelerating convergence for PDE solving tasks.

## Enviroment

The recommend enviroment is 
```
docker pull ghcr.io/nvidia/jax:equinox
```

## Usage

Execute the main script to train a PINN with quasi-sampling on a benchmark PDE (e.g., Poisson equation):

Training mode:  
```
python poisson.py --mode train --network mlp --dim 100
```
Evaluation mode:
```
python poisson.py --mode eval --network mlp --dim 100
```

## Main results
With Allen_Cahn equations
| $d$  | **Vanilla**                             | **RAD**                                 | **Halton**                              | **Sobol**                               |
|------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| 3    | $3.89 \times 10^{-4} \pm 1.19 \times 10^{-4}$ | $4.30 \times 10^{-4} \pm 2.31 \times 10^{-4}$ | $2.44 \times 10^{-4} \pm 4.82 \times 10^{-5}$ | $3.39 \times 10^{-4} \pm 1.15 \times 10^{-4}$ |
| 5    | $3.81 \times 10^{-4} \pm 1.35 \times 10^{-5}$ | $4.49 \times 10^{-4} \pm 7.98 \times 10^{-5}$ | $4.07 \times 10^{-4} \pm 6.94 \times 10^{-6}$ | $4.27 \times 10^{-4} \pm 9.29 \times 10^{-6}$ |
| 10   | $9.61 \times 10^{-4} \pm 4.34 \times 10^{-5}$ | $9.08 \times 10^{-4} \pm 6.65 \times 10^{-5}$ | $8.64 \times 10^{-4} \pm 3.86 \times 10^{-6}$ | $9.21 \times 10^{-4} \pm 7.43 \times 10^{-5}$ |
| 20   | $2.15 \times 10^{-3} \pm 1.10 \times 10^{-4}$ | $2.12 \times 10^{-3} \pm 7.85 \times 10^{-5}$ | $2.10 \times 10^{-3} \pm 5.44 \times 10^{-5}$ | $2.06 \times 10^{-3} \pm 1.05 \times 10^{-4}$ |
| 50   | $6.95 \times 10^{-3} \pm 3.31 \times 10^{-4}$ | $7.58 \times 10^{-3} \pm 3.52 \times 10^{-4}$ | $7.02 \times 10^{-3} \pm 1.34 \times 10^{-4}$ | $6.93 \times 10^{-3} \pm 4.06 \times 10^{-4}$ |
| 80   | $1.43 \times 10^{-2} \pm 1.25 \times 10^{-4}$ | $1.43 \times 10^{-2} \pm 1.41 \times 10^{-4}$ | $1.29 \times 10^{-2} \pm 7.12 \times 10^{-4}$ | $1.42 \times 10^{-2} \pm 2.16 \times 10^{-4}$ |
| 90   | $1.59 \times 10^{-2} \pm 1.70 \times 10^{-4}$ | $1.60 \times 10^{-2} \pm 3.40 \times 10^{-4}$ | $1.28 \times 10^{-2} \pm 8.16 \times 10^{-5}$ | $1.58 \times 10^{-2} \pm 1.25 \times 10^{-4}$ |
| 100  | $1.69 \times 10^{-2} \pm 5.77 \times 10^{-5}$ | $1.69 \times 10^{-2} \pm 5.77 \times 10^{-5}$ | $1.30 \times 10^{-2} \pm 5.51 \times 10^{-4}$ | $1.71 \times 10^{-2} \pm 1.53 \times 10^{-4}$ |

## Citation

    @article{YU2026132913,
    title = {Quasi-random physics-informed neural networks},
    journal = {Neurocomputing},
    pages = {132913},
    year = {2026},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2026.132913},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231226003103},
    author = {Tianchi Yu and Ivan Oseledets},
}
