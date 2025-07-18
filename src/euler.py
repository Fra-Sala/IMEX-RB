import os
import numpy as np
import scipy.sparse
from src.newton import newton

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def forward_euler(problem, u0, tspan, Nt):
    """
    Implements the Forward Euler time integration scheme for solving ODEs.

    Parameters:
    -----------
    problem : object
        An object representing the problem to be solved.

    u0 : array-like
        The initial condition for the ODE system.

    tspan : tuple
        A tuple `(t0, tf)` specifying the initial and final times of the
        integration.

    Nt : int
        The number of time steps to use for the integration.

    Returns:
    --------
    u : ndarray
        The solution array of shape `(problem.Nh, Nt + 1)` containing the
        solution at all time steps. Returned only if full history allocation
        succeeds.

    un : ndarray
        The solution at the final time step. Returned only if full history
        allocation fails.

    tvec : ndarray
        The time vector of shape `(Nt + 1,)` containing the time points
        corresponding to the solution.

    Notes:
    ------
    - Dirichlet boundary conditions are enforced at each time step using the
      indices provided by `problem.dirichlet_idx` and the values computed by
      `problem.lift_vals`.
    """

    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt + 1)

    try:
        u = np.zeros((problem.Nh, Nt + 1))
        u[:, 0] = u0
        un = u0.copy()
        save_all = True
    except MemoryError:
        logger.info("Memory issue for FE. Saving only un \n")
        un = u0.copy()
        save_all = False

    # Retrieve Dirichlet indices
    Didx = problem.dirichlet_idx

    for n in range(Nt):
        unp1 = un + dt * problem.rhs(tvec[n], un)
        # Enforce Dirichlet BCs
        unp1[Didx] = problem.lift_vals(tvec[n + 1])[Didx]

        if save_all:
            u[:, n + 1] = unp1

        un = unp1

    if save_all:
        return u, tvec

    return un, tvec


def backward_euler(problem, u0, tspan, Nt, solver="gmres", typeprec=None):
    """
    Implements the Backward Euler time integration scheme for solving ODEs.

    Parameters:
    -----------
    problem : object
        An object representing the problem to be solved.

    u0 : ndarray
        Initial condition vector.

    tspan : tuple
        A tuple `(t0, tf)` specifying the initial and final times.

    Nt : int
        Number of time steps.

    solver : str, optional
        Solver to use for the Newton method. Default is "gmres".

    typeprec : optional
        Type of preconditioner to use. Default is None.

    Returns:
    --------
    u : ndarray
        Solution array of shape `(problem.Nh, Nt + 1)` if memory allows.
        Returned only if `save_all` is True.

    un : ndarray
        Solution vector at the final time step.
        Returned only if `save_all` is False.

    tvec : ndarray
        Time vector of shape `(Nt + 1,)` containing the time points.

    Notes:
    ------
    - Dirichlet boundary conditions are enforced at each time step using the
    indices provided by `problem.dirichlet_idx` and the values computed by
    `problem.lift_vals`.
    """
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt + 1)

    try:
        u = np.zeros((problem.Nh, Nt + 1))
        u[:, 0] = u0
        save_all = True
    except MemoryError:
        logger.info("Memory issue for BE. Saving only un \n")
        un = u0.copy()
        save_all = False

    # Retrieve non-Dirichlet indices
    free_idx = problem.free_idx

    if problem.is_linear:
        jacF = scipy.sparse.identity(u0[free_idx].shape[0]) - \
               dt * problem.jacobian_free(tvec[0], u0)
        precM = problem.preconditioner(jacF, typeprec=typeprec)

    for n in range(Nt):
        # Define u(t_n)
        uold = u[:, n] if save_all else un
        uold0 = uold[free_idx]
        # Prepare unp1
        unp1 = np.zeros(np.shape(uold))

        # Solve for internal nodes only
        F = (lambda x: x - uold0 - dt * problem.rhs_free(tvec[n + 1], x))

        if not problem.is_linear:
            # Jacobian of F
            jacF = scipy.sparse.identity(uold0.shape[0]) - \
                dt * problem.jacobian_free(tvec[n + 1], uold)
            # Define preconditioner. Default is None
            precM = problem.preconditioner(jacF, typeprec=typeprec)

        # Solve
        unp1[free_idx], *_ = newton(F, jacF, uold0,
                                    solver=solver, option='qNewton',
                                    is_linear=problem.is_linear, prec=precM,
                                    tol=1e-3*dt**2)
        # Enforce BCs values
        unp1 += problem.lift_vals(tvec[n + 1])

        if save_all:
            u[:, n + 1] = unp1
        un = unp1

    if save_all:
        return u, tvec

    return un, tvec
