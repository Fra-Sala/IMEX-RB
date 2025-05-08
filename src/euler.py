import numpy as np
import time
import scipy


def forward_euler(problem, u0, tspan, Nt):
    start = time.time()
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt+1)
    u = np.zeros((problem.N, Nt+1))
    u[:, 0] = u0
    for n in range(Nt):
        # assemble rhs b
        b = problem.source_term(tvec[n])
        # We assemble all entries, even though the assembled
        # Dirichlet entries will be overwritten
        u[:, n+1] = u[:, n] + dt * (problem.A @ u[:, n] + b)
        u[:, n+1] = problem.enforce_bcs(u[:, n+1], tvec[n+1])
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
