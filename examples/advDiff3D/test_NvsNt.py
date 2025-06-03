
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
import numpy as np
import timeit

from src.problemsPDE import AdvDiff3D
from src.euler import backward_euler, forward_euler
from src.imexrb import imexrb
from utils.helpers import integrate_1D, cond_sparse, create_test_directory, \
    compute_steps_stability_FE
from utils.errors import compute_errors

from examples.advDiff3D.config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """
    In this test, we evaluate the performances of IMEX-RB, compared
    to those of backward Euler, considering different timestep values.
    We investigate the 3D advection diffusion problem
    In particular, we study:
    Convergence, average inner iter, and CPU time vs \Delta t.
    -- Multiple curves are produced varying N. --
    We will then also plot the iterations over time for a given
    \Delta t.
    """
    # Define test parameters
    N_values = [1, 5, 10, 20]  # minimal dimension of the reduced basis
    Nt_values = [2 ** n for n in range(4, 8)]  # range of Nt values

    n_solves = 0  # number of solver calls to robustly estimate computational times

    # Setup problem
    problem = AdvDiff3D(Nx, Ny, Nz, Lx, Ly, Lz, mu=mu,
                        sigma=sigma, vx=vx, vy=vy, vz=vz)

    # Define test directory
    testname = "NvsNt"
    test_dir = create_test_directory(os.path.join(results_dir, problem.name),
                                     testname)
    u0 = problem.initial_condition()

    epsilon = 0.003790879330088758  # N_i = 51, tol =1e-2 # 1.0 / cond_sparse(problem.A)  # epsilon for absolute stability condition
    logger.debug(f"Considering epsilon = {epsilon:.4e}")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((len(Nt_values), len(N_values))),
                 "BE": np.empty(len(Nt_values)),
                 "FE": np.empty(len(Nt_values))}
    errors_all = {"IMEX-RB": np.empty((len(Nt_values), len(N_values), Nt_values[-1])),
                  "BE": np.empty((len(Nt_values), Nt_values[-1])),
                  "FE": np.empty((len(Nt_values), Nt_values[-1]))}
    times = {"IMEX-RB": np.zeros((len(Nt_values), len(N_values))),
             "BE": np.zeros(len(Nt_values)),
             "FE": np.empty(len(Nt_values))}
    subiters = {"IMEX-RB": np.empty((len(Nt_values), len(N_values), Nt_values[-1])),
                "BE": None, "FE": None}

    Nt_FE = 248 # for Nx = Ny = Nz = 51 tol=1e-2 # compute_steps_stability_FE(problem, [t0, T])

    for cnt_Nt, _Nt in enumerate(Nt_values):
        print("\n")
        logger.info(f"Solving for Nt={_Nt}")
        tvec = np.linspace(t0, T, _Nt + 1)

        logger.info("Solving with Backward Euler (BE)")
        uBE, *_ = backward_euler(problem, u0, [t0, T], _Nt, **sparse_solver)

        if n_solves > 0:
            f_BE = lambda: backward_euler(problem, u0, [t0, T], _Nt, **sparse_solver)
            timer = timeit.Timer(f_BE)
            _t = timer.repeat(number=1, repeat=n_solves)
            times["BE"][cnt_Nt] += np.mean(_t)

        errors_all["BE"][cnt_Nt, :_Nt] = compute_errors(uBE, tvec, problem, mode="all")
        errors_l2["BE"][cnt_Nt] = integrate_1D(errors_all["BE"][cnt_Nt, :_Nt], tvec[1:])

        logger.info("Solving with Forward Euler (FE)")
        uFE, *_ = forward_euler(problem, u0, [t0, T], _Nt)

        if n_solves > 0:
            f_FE = lambda: forward_euler(problem, u0, [t0, T], _Nt)
            timer = timeit.Timer(f_FE)
            _t = timer.repeat(number=1, repeat=n_solves)
            times["FE"][cnt_Nt] += np.mean(_t)

        errors_all["FE"][cnt_Nt, :_Nt] = compute_errors(uFE, tvec, problem, mode="all")
        errors_l2["FE"][cnt_Nt] = integrate_1D(errors_all["FE"][cnt_Nt, :_Nt], tvec[1:])

        logger.info("Solving with IMEX-RB")

        for cnt_N, N in enumerate(N_values):
            logger.info(f"Solving for N={N}")

            uIMEX, _, iters = imexrb(problem, u0, [t0, T], _Nt, epsilon, N, maxsubiter)

            if n_solves > 0:
                f_IMEX = lambda: imexrb(problem, u0, [t0, T], _Nt, epsilon, N, maxsubiter)
                timer = timeit.Timer(f_IMEX)
                _t = timer.repeat(number=1, repeat=n_solves)
                times["IMEX-RB"][cnt_Nt, cnt_N] += np.mean(_t)

            # Store subiterates
            subiters["IMEX-RB"][cnt_Nt, cnt_N, :_Nt] = iters

            # Compute errors
            errors_all["IMEX-RB"][cnt_Nt, cnt_N, :_Nt] = compute_errors(uIMEX, tvec, problem,  mode="all")
            errors_l2["IMEX-RB"][cnt_Nt, cnt_N] = integrate_1D(errors_all["IMEX-RB"][cnt_Nt, cnt_N, :_Nt], tvec[1:])

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             Nt_FE=np.array([Nt_FE]),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             subiters=subiters,
             N_values=N_values,
             Nt_values=Nt_values,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
