import os
import numpy as np
import logging.config
from utils.helpers import integrate_1D

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def compute_errors(u, tvec, problem, q=2, mode="all"):
    """
    Compute the relative errors between the numerical and exact solutions
    for a given problem over time.

    Parameters:
    -----------
    u : ndarray
        Numerical solution array with dimensions corresponding to the
        spatial and temporal discretization.
    tvec : ndarray
        Array of time points at which the solution is evaluated.
    problem : object
        Problem object for FD discretization.
    q : float, optional
        Order of the norm used to compute the errors. Defaults to 2 (L2 norm).
        Use `np.inf` for the infinity norm.
    mode : str, optional
        If 'all', errors at all times are returned;
        If 'l2', integral of the errors over time is returned
        If 'T', error at final time is returned

    Returns:
    --------
    errs : ndarray
        Array of relative errors for each solution component and time step.
        If `soldim > 1`, the shape is (soldim, nsteps), where `nsteps` is the
        number of time steps. If `soldim == 1`, the result is a 1D array of
        length `nsteps`.

    Notes:
    ------
    - The relative error is computed in a similar manner to Leveque,
    "Finite Difference Methods for Ordinary and Partial Differential
    Equations", 2007.
    """

    soldim = problem.soldim
    coords = problem.coords
    dxs = problem.dx
    exact = problem.exact_solution

    if mode == "T":
        tvec = tvec[-2:]
        u = u[..., -1]

    nsteps = len(tvec) - 1
    err_norms, sol_norms = np.empty((soldim, nsteps)), \
        np.empty((soldim, nsteps))
    volume = np.prod(dxs)

    for i in range(1, len(tvec)):
        t = tvec[i]
        u_flat = (u[..., i] if mode != "T" else u).ravel()
        npts = u_flat.size // soldim
        u_ex_all = exact(t, *coords).ravel()

        for c in range(soldim):
            start = c * npts
            end = start + npts
            u_num_c = u_flat[start:end]
            u_ex_c = u_ex_all[start:end]
            err = u_ex_c - u_num_c

            if np.isinf(q):
                err_norms[c, i - 1] = np.max(np.abs(err))
                sol_norms[c, i - 1] = np.max(np.abs(u_ex_c))
            elif q == -1:
                # Compute the error 'energy'
                err_norms[c, i - 1] = np.linalg.norm(err) ** 2
            else:
                err_norms[c, i - 1] = \
                    (volume * np.sum(np.abs(err) ** q)) ** (1 / q)
                sol_norms[c, i - 1] = \
                    (volume * np.sum(np.abs(u_ex_c) ** q)) ** (1 / q)

    if soldim == 1:
        err_norms, sol_norms = err_norms.ravel(), sol_norms.ravel()

    if mode == "all":  # Return errors over time
        errs_k = err_norms / sol_norms

    elif mode == "l2":  # Compute pseudo-integral over time
        # When the error has multiple components, set the axis for integration
        axis = 0 if soldim == 1 else 1
        err_norms = integrate_1D(err_norms, tvec[1:],
                                 method='midpoint', axis=axis)
        sol_norms = integrate_1D(sol_norms, tvec[1:],
                                 method='midpoint', axis=axis)
        errs_k = err_norms / sol_norms

    if soldim == 1:
        return errs_k  # One component for the solution
    else:
        return np.linalg.norm(errs_k, 2, axis=0)
