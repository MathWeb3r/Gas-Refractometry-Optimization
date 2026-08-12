# Gas Refractometry Optimize

> **Note:** This repository is part of an Undergraduate Research (*Iniciação Científica*) and scientific research project. It is a *work in progress*.

This project implements various methods to solve the inverse problem in gas refractometry, including non-constrained least squares, constrained optimization, and Tikhonov regularization. The goal is to determine the molar fractions of a gas mixture based on refractive index measurements at different wavelengths.

## Features

- **Multiple Solver Methods**:
  - **Normal**: Basic least squares inversion.
  - **LSTSQ**: Uses `numpy.linalg.lstsq` for better numerical stability.
  - **Constrain**: Uses `scipy.optimize.minimize` with constraints ($0 \le x_i \le 1$ and $\sum x_i = 1$).
  - **Tikhonov Regularization**: Implements the L-curve method to find the optimal regularization parameter (alpha).
  - **Bayesian Inference**: Implements Bayesian Inference to determine the molar fractions of the gas mixture.

## Usage & Tutorials

To learn how to use this repository, please check the `notebooks/` directory. It contains Jupyter Notebooks with detailed explanations and step-by-step examples of how to utilize the different methods implemented in this project.

