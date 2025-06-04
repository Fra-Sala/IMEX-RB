"""A configuration file for the advdiff3D example
(advection diffusion in 3D, cube domain)."""

import os

Lx = 2  # length in x-direction
Ly = 2  # length in y-direction
Lz = 2  # length in z-direction

Nx = 51   # number of elements in x-direction
Ny = 51  # number of elements in y-direction
Nz = 51   # number of elements in z-direction
Nt = 100  # number of time steps

mu = 0.01          # diffusion coefficient
sigma = 0.5

t0 = 0.0  # initial time
T = 1.0  # final time

vx = 1  # advection velocity in x-direction
vy = 0.5  # advection velocity in y-direction
vz = 0.5  # advection velocity in z-direction

N = 10           # number of bases
maxsubiter = 10  # maximal number of subiteration
eps = 1e-4   # absolute stability condition for the IMEX-RB method

sparse_solver = {"solver": "gmres", "typeprec": None}  # sparse solver
#############################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "__RESULTS")
os.makedirs(results_dir, exist_ok=True)
