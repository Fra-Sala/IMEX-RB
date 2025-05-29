"""A configuration file for the heat2D example (heat equation in 2D rectangular domain)."""

import os

Lx = 2  # length in x-direction
Ly = 2  # length in y-direction

Nx = 101   # number of elements in x-direction
Ny = 101   # number of elements in y-direction
Nt = 100   # number of time steps

mu = 0.02            # diffusion coefficient
sigma = 0.5          # decay in the exact solution
center = [0.5, 0.5]  # center of the exact solution exponential

vx = 1.0  # advection velocity in x-direction
vy = 0.5  # advection velocity in y-direction

t0 = 0.0  # initial time
T = 1.0   # final time

N = 10           # number of bases
maxsubiter = 10  # maximal number of subiterations
epsilon = 1e-4   # absolute stability condition for the IMEX-RB method

sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver configuration

#######################################################################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "results")
os.makedirs(results_dir, exist_ok=True)
