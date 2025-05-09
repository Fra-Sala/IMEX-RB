import scipy
import numpy as np


def compute_steps_stability_FE(problem, tspan, factor=0.95):
    """
    Compute minimum number of timesteps to make forward Euler (FE)
    scheme absolutely stable.
    """
    eigvals, _ = scipy.sparse.linalg.eigs(problem.A, k=1, which="LM")
    max_eig = abs(eigvals[0])
    dtFE = factor*2 / max_eig  # use 90% of the stability limit
    Nt_FE = int(np.ceil((tspan[1]-tspan[0]) / dtFE)) + 1
    return Nt_FE
