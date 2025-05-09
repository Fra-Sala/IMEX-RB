import numpy as np
import time
import scipy


def forward_euler(problem, u0, tspan, Nt):
    """
    Forward Euler time integration scheme. Because of the stability requirement
    it might not be possible to save an array of solutions for each time and each 
    position in space. In such a case, only un and unp1 are saved, and the solution
    at final time is returned.
    """
    start = time.time()
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt+1)
    try:
        u = np.zeros((problem.N, Nt+1))
        u[:, 0] = u0
        un = u0
        save_all = True
    except MemoryError:
        # Error: requested too much memory
        # Save only current solution
        un = u0
    
    for n in range(Nt):
        # assemble rhs b
        b = problem.source_term(tvec[n])
        # We assemble all entries, even though the assembled
        # Dirichlet entries will be overwritten
        unp1 = un + dt * (problem.A @ un + b)
        unp1 = problem.enforce_bcs(unp1, tvec[n+1])
        if save_all: u[..., n+1] = unp1
        un = unp1
    return u, tvec, time.time() - start


def backward_euler(problem, u0, tspan, Nt):
    start = time.time()
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt+1)
    u = np.zeros((problem.N, Nt+1))
    u[:, 0] = u0
    M = scipy.sparse.identity(problem.N, format='csr') - dt * problem.A
    for n in range(Nt):
        rhs = u[:, n] + dt * problem.source_term(tvec[n+1])
        # apply BC lifting to M and rhs
        M_bc, rhs_bc = problem.apply_bc(M, rhs, tvec[n+1])
        u[:, n+1] = scipy.sparse.linalg.spsolve(M_bc, rhs_bc)
    return u, tvec, time.time() - start
