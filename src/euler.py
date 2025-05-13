import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


def backward_euler(problem, u0, tspan, Nt, gmres=True):
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

    M = sparse.identity(problem.Nh, format='csr') - dt * problem.A
    # Get rid of Dirichlet rows and columns
    Mmod = problem.modify_system_matrix(M)
    Inodes = problem.non_dirichlet_indices

    for n in range(Nt):
        # Define u(t_n)
        uold = (u[:, n] if save_all else un)
        # Prepare unp1
        unp1 = np.zeros(np.shape(uold))
        # Assemble rhs, take into account BCs
        b = dt * problem.source_term(tvec[n + 1])
        bmod, uL = problem.apply_lifting(M, b, tvec[n + 1])
        rhs = uold[Inodes] + bmod
        # Solve sparse linear system
        if gmres:
            unp1[Inodes], *_ = \
                sparse.linalg.gmres(Mmod, rhs, M=problem.preconditioner(Mmod))
        else:
            unp1[Inodes] = spsolve(Mmod, rhs)
        # Enforce lifting values
        unp1 += uL
        if save_all:
            u[:, n + 1] = unp1
        un = unp1

    elapsed = time.time() - start
    if save_all:
        return u, tvec, elapsed
    return un, tvec, elapsed
