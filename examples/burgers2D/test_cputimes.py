import os
import sys
import numpy as np
# TO DO: fix import hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, cond_sparse, \
    create_test_directory
from utils.errors import compute_errors

from config import Lx, Ly, mu, t0, T, Nt, N, maxsubiter, \
    results_dir

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """We measure CPU times of IMEX-RB, BE and FE applied to the nonlinear
    2D Burgers equation, varying the size of the problem Nh"""

    Nx_values = [2 ** n for n in range(5, 11)]  # range of Nx values
    Nh_values = []
    n_solves = 10  # number of solver calls to robustly estimate times

    # Define test directory
    testname = "CPUtimes"
    test_dir = create_test_directory(os.path.join(results_dir, "Burgers2D"),
                                     testname)

    epsilon = 1e-4  # 1.0 / cond_sparse(problem.jacobian(t0, u0))  # epsilon guess 
    logger.debug(f"Considering epsilon = {epsilon}")

    # Initialise variables to track method performances
    soldim = 2
    errors_l2 = {"IMEX-RB": np.empty((soldim, len(Nx_values))),
                 "BE": np.empty((soldim, len(Nx_values)))}
    errors_all = {"IMEX-RB": np.empty((soldim, len(Nx_values),
                                       Nt)),
                  "BE": np.empty((soldim, len(Nx_values),
                                  Nt))}
    times = {"IMEX-RB": np.zeros(len(Nx_values)),
             "BE": np.zeros(len(Nx_values))}
    subiters = {"IMEX-RB": np.empty((len(Nx_values), Nt)),
                "BE": None}

    print("\n")
    logger.info(f"Solving for Nt={Nt}")
    tvec = np.linspace(t0, T, Nt + 1)

    for cnt_Nx, Nx in enumerate(Nx_values):

        Ny = Nx  # same discretization in x and y
        # Setup problem
        problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)
        u0 = problem.initial_condition()
        # Save problem size
        Nh_values.append(problem.Nh)
        logger.info(f"Solving for Nh={Nh_values[-1]}")

        logger.info("Solving with Backward Euler")
        for _ in range(n_solves):
            uBE, *_, _t = cpu_time(backward_euler, problem, u0,
                                   [t0, T], Nt, solver="gmres")
            times["BE"][cnt_Nx] += _t / n_solves

        errors_all["BE"][:, cnt_Nx, :Nt] = compute_errors(uBE, tvec, problem,
                                                          mode="all")
        errors_l2["BE"][:, cnt_Nx] = integrate_1D(
            errors_all["BE"][:, cnt_Nx, :Nt], tvec[1:], axis=1)

        logger.info("Solving with IMEX-RB")

        logger.info(f"Solving for N={N}")

        for _ in range(n_solves):
            uIMEX, *_, iters, _t = cpu_time(imexrb, problem, u0, [t0, T],
                                            Nt, epsilon, N, maxsubiter)
            times["IMEX-RB"][cnt_Nx] += _t / n_solves

        # Store subiterates
        subiters["IMEX-RB"][cnt_Nx, :Nt] = iters

        # Compute errors
        errors_all["IMEX-RB"][:, cnt_Nx, :Nt] = compute_errors(uIMEX, tvec,
                                                               problem,
                                                               mode="all")
        errors_l2["IMEX-RB"][:, cnt_Nx] = integrate_1D(
            errors_all["IMEX-RB"][:, cnt_Nx, :Nt], tvec[1:], axis=1)

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             subiters=subiters,
             N_values=N,
             Nh_values=Nh_values,
             maxsubiter=maxsubiter,
             epsilon=epsilon,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
