"""A configuration file for the burgers2D example
(viscous Burgers equation in 2D over a square domain)."""

import os

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction

Nx = 101   # number of elements in x-direction
Ny = 101   # number of elements in y-direction
Nt = 40  # number of time steps

mu = 0.01          # diffusion coefficient

t0 = 0.0  # initial time
T = 1.0  # final time

N = 10  # number of bases
maxsubiter = 30  # maximal number of subiterations
sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver 
#############################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "__RESULTS")
os.makedirs(results_dir, exist_ok=True)
