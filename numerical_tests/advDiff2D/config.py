"""A configuration file for the heat2D example (heat equation in 2D rectangular domain)."""

import os

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction

Ni_values = [101, 201]  # TODO
# Nx = 101   # number of elements in x-direction
# Ny = 101   # number of elements in y-direction
Nt = 100   # number of time steps

mu = 0.005            # diffusion coefficient
sigma = 0.25         # decay in the exact solution
center = [0.25, 0.25]  # center of the exact solution exponential

vx = 0.5  # advection velocity in x-direction
vy = 0.25  # advection velocity in y-direction

t0 = 0.0  # initial time
T = 1.0   # final time

N = 10           # number of bases
maxsubiter = 100  # maximal number of subiterations
tol_cond_NtFE = 1e-8  # tolerance to compute sing values to estimate K(A)
sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver configuration

#######################################################################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "__RESULTS")
os.makedirs(results_dir, exist_ok=True)
