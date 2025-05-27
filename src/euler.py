import os
import numpy as np
import scipy.sparse
from src.newton import newton
from utils.helpers import get_linear_solver

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def forward_euler(problem, u0, tspan, Nt):
    """
    Forward Euler time integration scheme with memory fallback.

    If full history allocation fails, only current and next solution are saved,
    and the final solution is returned.
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
    Backward Euler time integration scheme with memory fallback.

    If full history allocation fails, only current and next solution
    are saved, and the final solution is returned.
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

    # linear_solver = get_linear_solver(solver=solver)

    for n in range(Nt):
        # Define u(t_n)
        uold = u[:, n] if save_all else un
        uold0 = uold[free_idx]
        # Prepare unp1
        unp1 = np.zeros(np.shape(uold))

        # Solve for internal nodes only
        F = (lambda x: x - uold0 - dt * problem.rhs_free(tvec[n + 1], x))

        # Jacobian of F
        jacF = scipy.sparse.identity(uold0.shape[0]) - \
            dt * problem.jacobian_free(tvec[n + 1], uold)
        # Define preconditioner. Default is None
        precM = problem.preconditioner(jacF, typeprec=typeprec)
        # Solve
        unp1[free_idx], *_ = newton(F, jacF, uold0,
                                    solver=solver, option='qNewton',
                                    is_linear=problem.is_linear, prec=precM)
        # Enforce BCs values
        unp1 += problem.lift_vals(tvec[n + 1])

        if save_all:
            u[:, n + 1] = unp1
        un = unp1

    if save_all:
        return u, tvec

    return un, tvec
