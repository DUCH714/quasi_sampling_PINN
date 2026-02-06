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
