"""A configuration file for the heat2D example (heat equation in 2D rectangular domain)."""

import os

Lx = 2  # length in x-direction
Ly = 2  # length in y-direction

Nx = 51   # number of elements in x-direction
Ny = 51   # number of elements in y-direction
Nt = 100  # number of time steps

mu = 0.02          # diffusion coefficient
sigma = 0.1        # decay in the exact solution
center = [0.5, 0.5]  # center of the exact solution exponential

t0 = 0.0  # initial time
T = 1.0  # final time

N = 10           # number of bases
maxsubiter = 30  # maximal number of subiterations
eps = 1e-4   # absolute stability condition for the IMEX-RB method

sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver configuration

#######################################################################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "results")
os.makedirs(results_dir, exist_ok=True)
