"""A configuration file for the advdiff3D example
(advection diffusion in 3D, cube domain)."""

import os

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction
Lz = 1  # length in z-direction

Nx = 101   # number of elements in x-direction
Ny = 101   # number of elements in y-direction
Nz = 101   # number of elements in z-direction
Nt = 40  # number of time steps

mu = 0.01          # diffusion coefficient
sigma = 0.5

t0 = 0.0  # initial time
T = 1.0  # final time

N = 10  # number of bases
maxsubiter = 30  # maximal number of subiterations

#############################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "results")
os.makedirs(results_dir, exist_ok=True)
