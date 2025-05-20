import numpy as np


def compute_errors(u, tvec, problem, q=2, finaltimeonly=False):
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
    finaltimeonly : bool, optional
        If True, compute errors only at the final time step. Defaults to False.

    Returns:
    --------
    errs : ndarray
        Array of relative errors for each solution component and time step.
        If `soldim > 1`, the shape is (soldim, nsteps), where `nsteps` is the 
        number of time steps. If `soldim == 1`, the result is a 1D array of 
        length `nsteps`.

    Notes:
    ------
    - The relative error is computed according to Leveque, 
    """
    soldim = problem.soldim
    coords = problem.coords
    dxs = problem.dx
    exact = problem.exact_solution

    if finaltimeonly:
        tvec = tvec[-2:]
        u = u[..., -1]

    nsteps = len(tvec) - 1
    errs = np.zeros((soldim, nsteps))
    volume = np.prod(dxs)

    for i in range(1, len(tvec)):
        t = tvec[i]
        u_flat = (u[..., i] if not finaltimeonly else u).ravel()
        npts = u_flat.size // soldim
        u_ex_all = exact(t, *coords).ravel()

        for c in range(soldim):
            start = c * npts
            end = start + npts
            u_num_c = u_flat[start:end]
            u_ex_c = u_ex_all[start:end]
            err = u_ex_c - u_num_c

            if np.isinf(q):
                errs[c, i - 1] = np.max(np.abs(err)) / np.max(np.abs(u_ex_c))
            else:
                norm_e = (volume * np.sum(np.abs(err) ** q)) ** (1 / q)
                norm_ex = (volume * np.sum(np.abs(u_ex_c) ** q)) ** (1 / q)
                errs[c, i - 1] = norm_e / norm_ex

    return errs if soldim > 1 else errs.ravel()
