"""A configuration file for the heat2D example (heat equation in 2D rectangular domain)."""

import os

problem_name = "Heat2D"

Lx = 1  # length in x-direction
Ly = 1  # length in y-direction

Nx = 50  # number of elements in x-direction
Ny = 50  # number of elements in y-direction

mu = 0.05  # diffusion coefficient

bc_left = 0.0    # BC datum on the left side
bc_right = 0.0   # BC datum on the right side
bc_bottom = 0.0  # BC datum on the bottom side
bc_top = 0.0     # BC datum on the top side

t0 = 0.0  # initial time
T = 1.0  # final time

#######################################################################################################################

project_path = os.getcwd()
results_dir = os.path.join(project_path, "results")
os.makedirs(results_dir, exist_ok=True)
