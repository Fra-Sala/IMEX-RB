"""A configuration file for the advdiff3D example
(advection diffusion in 3D, cube domain)."""

import os

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction
Lz = 1  # length in z-direction

Ni_values = [51, 81]  # Run simulations for different problem sizes
Nt = 100  # number of time steps

mu = 0.01          # diffusion coefficient
sigma = 0.25

t0 = 0.0  # initial time
T = 1.0  # final time

vx = 0.5  # advection velocity in x-direction
vy = 0.25  # advection velocity in y-direction
vz = 0.25  # advection velocity in z-direction
center = [0.25, 0.25, 0.25]
N = 10           # minimal number of bases
maxsubiter = 100  # maximal number of inner iterations
tol_cond_NtFE = 1e-2  # tolerance to compute sing values to estimate K(A)

sparse_solver = {"solver": "gmres", "typeprec": "ilu"}  # sparse solver
#############################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "__RESULTS")
os.makedirs(results_dir, exist_ok=True)
