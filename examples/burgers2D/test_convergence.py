import os
import sys
import numpy as np
# TO DO: fix import hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, \
    create_test_directory
from utils.errors import compute_errors

from config import Nx, Ny, Lx, Ly, mu, t0, T, N, maxsubiter, \
    results_dir

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """We check convergence of IMEX-RB applied to the nonlinear
    2D Burgers equation."""

    Nt_values = [2 ** n for n in range(3, 9)]  # range of Nt values

    n_solves = 1  # number of solver calls to robustly estimate times

    # Setup problem
    problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)

    # Define test directory
    testname = "convergence"
    test_dir = create_test_directory(os.path.join(results_dir, problem.name),
                                     testname)
    logger.debug(f"Running TEST: {testname}")
    u0 = problem.initial_condition()
    epsilon = 1e-4  # epsilon guess
    logger.debug(f"Considering epsilon = {epsilon}")
    logger.debug(f"Solving for N={N}")
    logger.debug(f"Solving for M={maxsubiter}")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((problem.soldim, len(Nt_values))),
                 "BE": np.empty((problem.soldim, len(Nt_values)))}
    errors_all = {"IMEX-RB": np.empty((problem.soldim, len(Nt_values),
                                       Nt_values[-1])),
                  "BE": np.empty((problem.soldim, len(Nt_values),
                                  Nt_values[-1]))}
    times = {"IMEX-RB": np.zeros(len(Nt_values)),
             "BE": np.zeros(len(Nt_values))}
    inneriters = {"IMEX-RB": np.empty((len(Nt_values), Nt_values[-1])),
                  "BE": None}

    for cnt_Nt, Nt in enumerate(Nt_values):
        print("\n")
        logger.info(f"Solving for Nt={Nt}")
        tvec = np.linspace(t0, T, Nt + 1)

        logger.info(f"Solving with BE for {n_solves} times")
        for _ in range(n_solves):
            uBE, *_, _t = cpu_time(backward_euler, problem, u0,
                                   [t0, T], Nt, solver="gmres")
            times["BE"][cnt_Nt] += _t / n_solves

        errors_all["BE"][:, cnt_Nt, :Nt] = compute_errors(uBE, tvec, problem,
                                                          mode="all")
        errors_l2["BE"][:, cnt_Nt] = integrate_1D(
            errors_all["BE"][:, cnt_Nt, :Nt], tvec[1:], axis=1)

        logger.info(f"Solving with IMEX-RB for {n_solves} times")
        for _ in range(n_solves):
            uIMEX, *_, iters, _t = cpu_time(imexrb, problem, u0, [t0, T],
                                            Nt, epsilon, N, maxsubiter)
            times["IMEX-RB"][cnt_Nt] += _t / n_solves

        logger.info(f"IMEX-RB performed inneriters (last run): "
                    f"avg={np.mean(iters)}, max={np.max(iters)}, "
                    f"tot={np.sum(iters)}")

        # Store subiterates
        inneriters["IMEX-RB"][cnt_Nt, :Nt] = iters

        # Compute errors
        errors_all["IMEX-RB"][:, cnt_Nt, :Nt] = compute_errors(uIMEX, tvec,
                                                               problem,
                                                               mode="all")
        errors_l2["IMEX-RB"][:, cnt_Nt] = integrate_1D(
            errors_all["IMEX-RB"][:, cnt_Nt, :Nt], tvec[1:], axis=1)

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             inneriters=inneriters,
             N_values=N,
             Nt_values=Nt_values,
             maxsubiter=maxsubiter,
             epsilon=epsilon,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
