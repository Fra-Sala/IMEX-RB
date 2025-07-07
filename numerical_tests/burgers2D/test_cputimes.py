import os
import sys
import numpy as np
import timeit
# TO DO: fix import hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import integrate_1D, \
    create_test_directory
from utils.errors import compute_errors

from config import Lx, Ly, mu, t0, T, Nt, \
    results_dir, sparse_solver

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """We measure CPU times of IMEX-RB, BE and FE applied to the nonlinear
    2D Burgers equation, varying the size of the problem Nh"""

    Nx_values = [2 ** n for n in range(5, 6)] # 10)]  # range of Nx values
    Nh_values = []
    N_values = [5, 10, 25]
    n_solves = 5  # number of solver calls to robustly estimate times

    # Define test directory
    testname = "CPUtimes"
    test_dir = create_test_directory(os.path.join(results_dir, "Burgers2D"),
                                     testname)

    epsilon = 1e-4  # Epsilon, justified by the other tests
    maxsubiter = 100  # Increased: N_h grows
    logger.debug(f"Running TEST: {testname}")
    logger.debug(f"Considering epsilon = {epsilon}")
    logger.debug(f"Solving for {len(N_values)} different N for IMEX-RB")
    logger.debug(f"Solving for M={maxsubiter}")

    # Initialise variables to track method performances
    errors_l2 = {"IMEX-RB": np.empty((len(N_values), len(Nx_values))),
                 "BE": np.empty((len(Nx_values)))}
    errors_all = {"IMEX-RB": np.empty((len(N_values), len(Nx_values),
                                       Nt)),
                  "BE": np.empty((len(Nx_values),
                                  Nt))}
    times = {"IMEX-RB": np.zeros((len(N_values), len(Nx_values))),
             "BE": np.zeros(len(Nx_values))}
    inneriters = {"IMEX-RB": np.empty((len(N_values), len(Nx_values), Nt)),
                  "BE": None}

    logger.debug(f"Solving for Nt={Nt}")
    tvec = np.linspace(t0, T, Nt + 1)

    # Start simulations
    for cnt_Nx, Nx in enumerate(Nx_values):

        Ny = Nx  # same discretization in x and y
        # Setup problem
        problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)
        u0 = problem.initial_condition()
        # Save problem size
        Nh_values.append(problem.Nh)
        print("\n")
        logger.info(f"Solving for Nh={Nh_values[-1]}")
        # Solving one time for errors
        uBE, *_ = backward_euler(problem, u0, [t0, T], Nt,
                                 **sparse_solver)
        logger.info(f"Solving with BE for {n_solves} times")
        # Repeating for CPU times
        if n_solves > 0:
            f_BE = (lambda: backward_euler(problem, u0, [t0, T], Nt,
                                           **sparse_solver))
            timer = timeit.Timer(f_BE)
            _t = timer.repeat(number=1, repeat=n_solves)
            times["BE"][cnt_Nx] += np.mean(_t)

        errors_all["BE"][cnt_Nx, :Nt] = compute_errors(uBE, tvec, problem,
                                                       mode="all")
        errors_l2["BE"][cnt_Nx] = compute_errors(uBE, tvec, problem,
                                                 mode="l2")

        for cnt_N, Nval in enumerate(N_values):
            logger.info(f"Solving for N={Nval}")
            # Solve one time for errors
            uIMEX, _, iters = imexrb(problem, u0, [t0, T], Nt, epsilon, Nval,
                                     maxsubiter)
            logger.info(f"Solving with IMEX-RB for {n_solves} times")
            # Repeat for CPU times
            if n_solves > 0:
                f_IMEX = (lambda: imexrb(problem, u0, [t0, T], Nt,
                                         epsilon, Nval, maxsubiter))
                timer = timeit.Timer(f_IMEX)
                _t = timer.repeat(number=1, repeat=n_solves)
                times["IMEX-RB"][cnt_N, cnt_Nx] += np.mean(_t)

            logger.info(f"IMEX-RB performed inneriters: "
                        f"avg={np.mean(iters)}, max={np.max(iters)}, "
                        f"tot={np.sum(iters)}")
            # Store subiterates
            inneriters["IMEX-RB"][cnt_N, cnt_Nx, :Nt] = iters

            # Compute errors
            errors_all["IMEX-RB"][cnt_N, cnt_Nx, :Nt] = \
                compute_errors(uIMEX, tvec, problem, mode="all")
            errors_l2["IMEX-RB"][cnt_N, cnt_Nx] = \
                compute_errors(uIMEX, tvec, problem, mode="l2")

    # Save results
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_l2=errors_l2,
             errors_all=errors_all,
             times=times,
             inneriters=inneriters,
             N_values=N_values,
             Nh_values=Nh_values,
             maxsubiter=maxsubiter,
             epsilon=epsilon,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
