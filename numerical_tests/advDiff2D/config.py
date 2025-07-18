"""A configuration file for the AdvDiff2D example
(advection diffusion equation in 2D square domain)."""

import os

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction

Ni_values = [101, 201]  # Values of Nx=Ny to be used for multiple sims
Nt = 100   # number of time steps

mu = 0.005           # diffusion coefficient
sigma = 0.25         # width of initial blob
center = [0.25, 0.25]  # center of the exact solution exponential

vx = 0.5  # advection velocity in x-direction
vy = 0.25  # advection velocity in y-direction

t0 = 0.0  # initial time
T = 1.0   # final time

N = 10           # number of bases
maxsubiter = 100  # maximal number of inner iterations
epsilon = 1e-4    # stability tolerance for main script
tol_cond_NtFE = 1e-8  # tolerance to compute sing values to estimate K(A)
sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver

##########################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "__RESULTS")
os.makedirs(results_dir, exist_ok=True)
