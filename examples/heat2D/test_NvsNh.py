import sys
sys.path.append('../..')

import os
import numpy as np

from src.problemsPDE import Heat2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, cond_sparse, create_test_directory, compute_steps_stability_FE
from utils.errors import compute_errors

from config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """In this test, we evaluate the performances of IMEX-RB, compared to those of backward Euler,
    considering different spatial discretizations."""

    # Define test parameters
    # N_values = np.array([5, 10, 15, 20, 25, 30])  # minimal dimension of the reduced basis
    # Nt_values = np.array([2 ** n for n in range(2, 12)])  # range of Nt values

    N_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # minimal dimension of the reduced basis
    Nh_values = np.array([10, 20, 40])  # range of Nh values

    update_Nt = True   # update the value of Nt as Nh changes

    n_solves = 1  # number of solver calls to robustly estimate computational times

    # Define test directory
    test_dir = create_test_directory(results_dir, "NvsNh")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((len(Nh_values), len(N_values))),
                 "BE": np.empty(len(Nh_values))}
    errors_all = {"IMEX-RB": np.empty((len(Nh_values), len(N_values), Nt)),
                  "BE": np.empty((len(Nh_values), Nt))}
    times = {"IMEX-RB": np.zeros((len(Nh_values), len(N_values))),
             "BE": np.zeros(len(Nh_values))}
    subiters = {"IMEX-RB": np.empty((len(Nh_values), len(N_values), Nt)),
                "BE": None}

    _Nt = Nt

    for cnt_Nh, Nh in enumerate(Nh_values):
        print("\n")
        logger.info(f"Solving for Nh={Nh}")

        if update_Nt:
            _Nt = compute_steps_stability_FE(problem, [t0, T], factor=20)

        tvec = np.linspace(t0, T, _Nt + 1)

        # Setup problem
        problem = Heat2D(Nh, Nh, Lx, Ly, mu=mu, sigma=sigma, center=center)
        u0 = problem.initial_condition()
        epsilon = 1.0 / cond_sparse(problem.A)  # epsilon for absolute stability condition --> OK since A is symmetric
        logger.debug(f"Considering epsilon = {epsilon:.4e}")

        logger.info("Solving with Backward Euler")
        for _ in range(n_solves):
            uBE, *_, _t = cpu_time(backward_euler, problem, u0, [t0, T], _Nt,
                                   solver="gmres", typeprec="ilu")

        times["BE"][cnt_Nh] += _t / n_solves
        errors_all["BE"][cnt_Nh] = compute_errors(uBE, tvec, problem, mode="all")
        errors_l2["BE"][cnt_Nh] = integrate_1D(errors_all["BE"][cnt_Nh], tvec[1:])

        logger.info("Solving with IMEX-RB")

        for cnt_N, N in enumerate(N_values):
            logger.info(f"Solving for N={N}")

            for _ in range(n_solves):
                uIMEX, *_, iters, _t = cpu_time(imexrb, problem, u0, [t0, T], _Nt,
                                                epsilon, N, maxsubiter)
                times["IMEX-RB"][cnt_Nh, cnt_N] += _t / n_solves

            # Store subiterates
            subiters["IMEX-RB"][cnt_Nh, cnt_N] = iters

            # Compute errors
            errors_all["IMEX-RB"][cnt_Nh, cnt_N] = compute_errors(uIMEX, tvec, problem, mode="all")
            errors_l2["IMEX-RB"][cnt_Nh, cnt_N] = integrate_1D(errors_all["IMEX-RB"][cnt_Nh, cnt_N], tvec[1:])

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             subiters=subiters,
             N_values=N_values,
             Nh_values=Nh_values,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
