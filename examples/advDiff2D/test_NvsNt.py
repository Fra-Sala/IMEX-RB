import sys
sys.path.append('../..')

import os
import numpy as np

from src.problemsPDE import AdvDiff2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, cond_sparse, create_test_directory
from utils.errors import compute_errors

from config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """In this test, we evaluate the performances of IMEX-RB, compared to those of backward Euler,
    considering different timestep values."""

    # Define test parameters
    # N_values = [5, 10, 15, 20, 25, 30]  # minimal dimension of the reduced basis
    # Nt_values = [2 ** n for n in range(2, 12)]  # range of Nt values

    N_values = [1,2,3,4,5,6,7,8,9,10]  # minimal dimension of the reduced basis
    Nt_values = [2 ** n for n in range(6,7)]  # range of Nt values

    n_solves = 1  # number of solver calls to robustly estimate computational times

    # Define test directory
    test_dir = create_test_directory(results_dir, "NvsNt")

    # Setup problem
    problem = AdvDiff2D(Nx, Ny, Lx, Ly, mu=mu, sigma=sigma, vx=vx, vy=vy, center=center)
    u0 = problem.initial_condition()

    epsilon = 1.0 / cond_sparse(problem.A)  # epsilon for absolute stability condition
    logger.debug(f"Considering epsilon = {epsilon:.4e}")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((len(Nt_values), len(N_values))),
                 "BE": np.empty(len(Nt_values))}
    errors_all = {"IMEX-RB": np.empty((len(Nt_values), len(N_values), Nt_values[-1])),
                  "BE": np.empty((len(Nt_values), Nt_values[-1]))}
    times = {"IMEX-RB": np.zeros((len(Nt_values), len(N_values))),
             "BE": np.zeros(len(Nt_values))}
    subiters = {"IMEX-RB": np.empty((len(Nt_values), len(N_values), Nt_values[-1])),
                "BE": None}

    for cnt_Nt, Nt in enumerate(Nt_values):
        print("\n")
        logger.info(f"Solving for Nt={Nt}")
        tvec = np.linspace(t0, T, Nt + 1)

        logger.info("Solving with Backward Euler")
        for _ in range(n_solves):
            uBE, *_, _t = cpu_time(backward_euler, problem, u0, [t0, T], Nt, solver="gmres", typeprec="ilu")

        times["BE"][cnt_Nt] += _t / n_solves
        errors_all["BE"][cnt_Nt, :Nt] = compute_errors(uBE, tvec, problem, mode="all")
        errors_l2["BE"][cnt_Nt] = integrate_1D(errors_all["BE"][cnt_Nt, :Nt], tvec[1:])

        logger.info("Solving with IMEX-RB")

        for cnt_N, N in enumerate(N_values):
            logger.info(f"Solving for N={N}")

            for _ in range(n_solves):
                uIMEX, *_, iters, _t = cpu_time(imexrb, problem, u0, [t0, T], Nt, epsilon, N, maxsubiter)
                times["IMEX-RB"][cnt_Nt, cnt_N] += _t / n_solves

            # Store subiterates
            subiters["IMEX-RB"][cnt_Nt, cnt_N, :Nt] = iters

            # Compute errors
            errors_all["IMEX-RB"][cnt_Nt, cnt_N, :Nt] = compute_errors(uIMEX, tvec, problem,  mode="all")
            errors_l2["IMEX-RB"][cnt_Nt, cnt_N] = integrate_1D(errors_all["IMEX-RB"][cnt_Nt, cnt_N, :Nt], tvec[1:])

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
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
