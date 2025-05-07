# FD-PyIMEX-RB

This project implements the IMEX-RB (Implicit-Explicit Reduced Basis) method to solve time-dependent problems discretized in space using finite difference (FD) schemes.

## Getting Started

To run the simulation and reproduce results, open and execute the Jupyter notebook:

- `main_imexrb.ipynb` — the main notebook for running simulations and visualizing results.

### Utilities
- `problems1D.py` - class of 1D problems, inheriting from a parent class.
- `imexrb.py` - the novel time integration method.
- `euler.py` - implementation of classic backward and forward Euler.
- `utils.py` — helper functions and utilities used by the notebook.

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
