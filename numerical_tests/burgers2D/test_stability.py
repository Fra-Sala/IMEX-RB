import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler, forward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, create_test_directory
from utils.errors import compute_errors

from config import Nx, Ny, Lx, Ly, mu, t0, T, Nt, N, maxsubiter, \
    results_dir

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """
    In this test, we evaluate the stability of IMEX-RB, compared
    to that of forward and backward Euler.
    We investigate the 2D Burgers' problem.
    In particular, we study:
    Accuracy and inner iterations over time.
    -- Multiple curves are produced varying $\\varepsilon$. --
    """

    # Solve only once (we are not interested in times here)
    n_solves = 1

    # Setup problem
    problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)

    # Define test directory
    testname = "stability"
    test_dir = create_test_directory(os.path.join(results_dir, problem.name),
                                     testname)

    u0 = problem.initial_condition()
    epsilon_values = [1e-2, 1e-3, 1e-4, 1e-5]
    logger.debug(f"Running TEST: {testname}")
    logger.debug(f"Solving for N={N}")
    logger.debug(f"Solving for M={maxsubiter}")
    logger.debug(f"Solving for Nt={Nt}")

    # Initialise variables to track method performances
    errors_stability = {"IMEX-RB": np.empty((len(epsilon_values),
                                             Nt)),
                        "BE": np.empty((Nt,)),
                        "FE": np.empty((Nt,))}
    times = {"IMEX-RB": np.zeros(len(epsilon_values), ),
             "BE": np.zeros(1,),
             "FE": np.zeros(1,)}
    inneriters = {"IMEX-RB": np.empty((len(epsilon_values), Nt)),
                  "BE": None,
                  "FE": None}

    print("\n")

    tvec = np.linspace(t0, T, Nt + 1)

    # BE and FE solutions are independent of epsilon
    logger.info(f"Solving with BE for {n_solves} times")
    for _ in range(n_solves):
        uBE, *_, _t = cpu_time(backward_euler, problem, u0,
                               [t0, T], Nt, solver="gmres")
        times["BE"] += _t / n_solves

    errors_stability["BE"] = compute_errors(uBE, tvec, problem,
                                            mode="all")
    logger.info(f"Solving with FE for {n_solves} times")
    for _ in range(n_solves):
        uFE, *_, _t = cpu_time(forward_euler, problem, u0,
                               [t0, T], Nt)
        times["FE"] += _t / n_solves

    errors_stability["FE"] = compute_errors(uFE, tvec, problem,
                                            mode="all")

    logger.info(f"Solving with IMEX-RB for {n_solves} times")
    for cnt_eps, epsilon in enumerate(epsilon_values):
        logger.info(f"Solving for epsilon = {epsilon}")

        for _ in range(n_solves):
            uIMEX, *_, iters, _t = cpu_time(imexrb, problem, u0, [t0, T],
                                            Nt, epsilon, N, maxsubiter)
            times["IMEX-RB"][cnt_eps] += _t / n_solves

        logger.info(f"IMEX-RB performed inneriters (last run): "
                    f"avg={np.mean(iters)}, max={np.max(iters)}, "
                    f"tot={np.sum(iters)}")

        # Store subiterates
        inneriters["IMEX-RB"][cnt_eps] = iters

        # Compute errors
        errors_stability["IMEX-RB"][cnt_eps] = compute_errors(uIMEX, tvec,
                                                              problem,
                                                              mode="all")
    np.savez(os.path.join(test_dir, "results.npz"),
             errors_stability=errors_stability,
             times=times,
             inneriters=inneriters,
             N_values=N,
             Nt=Nt,
             maxsubiter=maxsubiter,
             epsilon_values=epsilon_values,
             allow_pickle=True)

    return


if __name__ == "__main__":
    main()
