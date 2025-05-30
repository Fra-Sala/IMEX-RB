import sys
sys.path.append('../..')

import os
import numpy as np
import timeit

from src.problemsPDE import Heat2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import integrate_1D, cond_sparse, create_test_directory
from utils.errors import compute_errors

from examples.heat2D.config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """In this test, we evaluate the performances of IMEX-RB, compared to those of backward Euler,
    considering different timestep values."""

    # Define test parameters
    eps_values = np.array([10, 5, 1])  # values of epsilon for IMEX-RB, compared to ref
    Nt_values = np.array([2 ** n for n in range(6,8)])  # range of Nt values

    n_solves = 5  # number of solver calls to robustly estimate computational times

    # Define test directory
    test_dir = create_test_directory(results_dir, "EPSvsNt")

    # Setup problem
    problem = Heat2D(Nx, Ny, Lx, Ly, mu=mu, sigma=sigma, center=center)

    u0 = problem.initial_condition()
    epsilon_ref = 1.0 / cond_sparse(problem.A)  # epsilon for absolute stability condition
    logger.debug(f"Considering reference epsilon = {epsilon_ref:.4e}")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((len(Nt_values), len(eps_values))),
                 "BE": np.empty(len(Nt_values))}
    errors_all = {"IMEX-RB": np.empty((len(Nt_values), len(eps_values), Nt_values[-1])),
                  "BE": np.empty((len(Nt_values), Nt_values[-1]))}
    times = {"IMEX-RB": np.zeros((len(Nt_values), len(eps_values))),
             "BE": np.zeros(len(Nt_values))}
    subiters = {"IMEX-RB": np.empty((len(Nt_values), len(eps_values), Nt_values[-1])),
                "BE": None}

    for cnt_Nt, Nt in enumerate(Nt_values):
        print("\n")
        logger.info(f"Solving for Nt={Nt}")
        tvec = np.linspace(t0, T, Nt + 1)

        logger.info("Solving with Backward Euler")
        uBE, *_ = backward_euler(problem, u0, [t0, T], Nt, **sparse_solver)

        if n_solves > 0:
            f_BE = lambda: backward_euler(problem, u0, [t0, T], Nt, **sparse_solver)
            timer = timeit.Timer(f_BE)
            _t = timer.repeat(number=1, repeat=n_solves)
            times["BE"][cnt_Nt] += np.mean(_t)

        errors_all["BE"][cnt_Nt, :Nt] = compute_errors(uBE, tvec, problem, mode="all")
        errors_l2["BE"][cnt_Nt] = integrate_1D(errors_all["BE"][cnt_Nt, :Nt], tvec[1:])

        logger.info("Solving with IMEX-RB")

        for cnt_eps, eps in enumerate(eps_values):
            epsilon = epsilon_ref * eps
            logger.info(f"Solving for epsilon={epsilon:.4e}")

            uIMEX, _, iters = imexrb(problem, u0, [t0, T], Nt, epsilon, N, maxsubiter)

            if n_solves > 0:
                f_IMEX = lambda: imexrb(problem, u0, [t0, T], Nt, epsilon, N, maxsubiter)
                timer = timeit.Timer(f_IMEX)
                _t = timer.repeat(number=1, repeat=n_solves)
                times["IMEX-RB"][cnt_Nt, cnt_eps] += np.mean(_t)

            # Store subiterates
            subiters["IMEX-RB"][cnt_Nt, cnt_eps, :Nt] = iters

            # Compute errors
            errors_all["IMEX-RB"][cnt_Nt, cnt_eps, :Nt] = compute_errors(uIMEX, tvec, problem,  mode="all")
            errors_l2["IMEX-RB"][cnt_Nt, cnt_eps] = integrate_1D(errors_all["IMEX-RB"][cnt_Nt, cnt_eps, :Nt], tvec[1:])

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             subiters=subiters,
             eps_values=eps_values * epsilon_ref,
             Nt_values=Nt_values,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
