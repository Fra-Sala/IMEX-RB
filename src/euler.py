import numpy as np


def forward_euler(problem, u0, tspan, Nt):
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt+1)
    u = np.zeros((problem.N, Nt+1))
    u[:, 0] = u0
    for n in range(Nt):
        u[:, n+1] = u[:, n] + dt * problem.rhs(tvec[n], u[:, n])
    return u, tvec


def backward_euler(problem, u0, tspan, Nt):
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt+1)
    u = np.zeros((problem.N, Nt+1))
    u[:, 0] = u0
    M = np.eye(problem.N) - dt * problem.A
    for n in range(Nt):
        rhs = u[:, n] + dt * problem.b(tvec[n+1])
        u[:, n+1] = np.linalg.solve(M, rhs)
    return u, tvec
