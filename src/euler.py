import numpy as np
import time
import scipy
from newton import newton


def forward_euler(problem, u0, tspan, Nt):
    """
    Forward Euler time integration scheme with memory fallback.

    If full history allocation fails, only current and next solution are saved,
    and the final solution is returned.
    """
    start = time.time()
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt + 1)
    try:
        u = np.zeros((problem.Nh, Nt + 1))
        u[:, 0] = u0
        un = u0.copy()
        save_all = True
    except MemoryError:
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

    elapsed = time.time() - start
    if save_all:
        return u, tvec, elapsed
    return un, tvec, elapsed


def backward_euler(problem, u0, tspan, Nt, solverchoice="gmres"):
    """
    Backward Euler time integration scheme with memory fallback.

    If full history allocation fails, only current and next solution
    are saved, and the final solution is returned.
    """
    start = time.time()
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt + 1)
    try:
        u = np.zeros((problem.Nh, Nt + 1))
        u[:, 0] = u0
        save_all = True
    except MemoryError:
        un = u0.copy()
        save_all = False
    # Retrieve non-Dirichlet indices
    free_idx = problem.free_idx

    for n in range(Nt):
        # Define u(t_n)
        uold = (u[:, n] if save_all else un)
        uold0 = uold[free_idx]
        # Prepare unp1
        unp1 = np.zeros(np.shape(uold))

        # Assemble nonlinear equation
        def F(x):
            """ Look for homogenous unp1_0  """
            return x - uold0 - dt*problem.rhs_free(tvec[n + 1], x)
        # Jacobian of F
        jacF = scipy.sparse.identity(uold0.shape[0]) - \
            dt * problem.jacobian_free(tvec[n + 1], uold)
        # Solve for internal nodes only
        unp1[free_idx], *_ = newton(F, jacF, uold0,
                                    solverchoice=solverchoice,
                                    option='qNewton')
        # Enforce BCs values
        unp1 += problem.lift_vals(tvec[n + 1])
        if save_all:
            u[:, n + 1] = unp1
        un = unp1

    elapsed = time.time() - start
    if save_all:
        return u, tvec, elapsed
    return un, tvec, elapsed
