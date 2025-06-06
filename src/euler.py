import os
import numpy as np
import scipy.sparse
from src.newton import newton

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

import psutil
import logging

logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S")
)
logger.addHandler(handler)


PID = os.getpid()
proc = psutil.Process(PID)
def log_memory(step_label=""):
    rss_bytes = proc.memory_info().rss
    print(f"[Memory] {step_label:30s} RSS = {rss_bytes / (1024**2):8.1f} MiB", flush=True)


np.seterr(all='raise')

def forward_euler(problem, u0, tspan, Nt):
    t0, tf = tspan
    dt = (tf - t0) / Nt
    tvec = np.linspace(t0, tf, Nt + 1)

    # Attempt to allocate full history
    log_memory("Before allocating u array")
    try:
        u = np.zeros((problem.Nh, Nt + 1))   #  OOM?
        log_memory("After allocating u array")
        u[:, 0] = u0
        un = u0.copy()
        save_all = True
    except MemoryError:
        log_memory("Caught MemoryError for u array")
        logger.info("Memory issue for FE. Saving only un\n")
        un = u0.copy()
        save_all = False

    Didx = problem.dirichlet_idx

    # Time‐loop (with memory checks)
    for n in range(Nt):
        if n % 50 == 0:
            log_memory(f"Before RHS at step {n:3d}")
        try:
            rhs_vec = problem.rhs(tvec[n], un)
        except MemoryError:
            log_memory(f"Caught MemoryError inside rhs() at step {n:3d}")
            raise
        log_memory(f"After rhs() at step {n:3d}")

        unp1 = un + dt * rhs_vec
        log_memory(f"After forming unp1 at step {n:3d}")

        # Enforce Dirichlet BCs
        bc_vals = problem.lift_vals(tvec[n + 1])
        unp1[Didx] = bc_vals[Didx]
        if save_all:
            u[:, n + 1] = unp1

        un = unp1

    if save_all:
        return u, tvec
    else:
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
