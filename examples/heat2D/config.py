"""A configuration file for the heat2D example (heat equation in 2D rectangular domain)."""

import os

Lx = 4  # length in x-direction
Ly = 4  # length in y-direction

Nx = 40   # number of elements in x-direction
Ny = 40   # number of elements in y-direction
Nt = 100  # number of time steps

mu = 1.5          # diffusion coefficient
sigma = 0.5        # decay in the exact solution
center = [2, 2]  # center of the exact solution exponential

t0 = 0.0  # initial time
T = 1.0  # final time

N = 10  # number of bases

#######################################################################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "results")
os.makedirs(results_dir, exist_ok=True)
