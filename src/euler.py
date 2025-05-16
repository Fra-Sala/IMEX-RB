import numpy as np
import time
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
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

    for n in range(Nt):
        b = problem.source_term(tvec[n])
        unp1 = un + dt * (problem.A @ un + b)
        unp1 = problem.compute_bcs(unp1, tvec[n + 1])
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
    Dindx = problem.dirichlet_indices

    for n in range(Nt):
        # Define u(t_n)
        uold = (u[:, n] if save_all else un)
        uold0 = uold.copy() 
        uold0[Dindx] = 0.0  # with 0 in Dindices
        uL = problem.compute_bcs(tvec[n + 1])
        # Prepare unp1
        unp1 = np.zeros(np.shape(uold))

        # Assemble nonlinear equation
        def F(x):
            return x + uL - uold - dt*problem.rhs(tvec[n + 1], x + uL)
        # jacobian of F
        jacF = scipy.sparse.identity(u0.shape[0]) - \
            dt*problem.jacobian(tvec[n], uold)
        unp1, *_ = newton(problem, Dindx, F, jacF, uold0,
                          solverchoice, option='qNewton')
        # Enforce BCs values
        unp1[Dindx] = uL[Dindx]
        if save_all:
            u[:, n + 1] = unp1
        un = unp1

    elapsed = time.time() - start
    if save_all:
        return u, tvec, elapsed
    return un, tvec, elapsed
