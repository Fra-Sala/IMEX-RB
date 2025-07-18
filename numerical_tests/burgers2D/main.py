import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))

import numpy as np
from tabulate import tabulate
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time
from utils.errors import compute_errors

from numerical_tests.burgers2D.config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """Main file to run one simulation of Burgers 2D
    problem, comparing backward Euler and IMEX-RB times and accuracy."""
    # Same discretization along all directions
    Ny = Nx
    # Setup problem
    problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)
    u0 = problem.initial_condition()

    logger.debug(f"Considering epsilon = {epsilon:.4e}")

    print("\n")
    tvec = np.linspace(t0, T, Nt + 1)

    logger.info("Solving with Backward Euler")
    uBE, *_, _tBE = cpu_time(backward_euler, problem, u0, [t0, T], Nt,
                             **sparse_solver)
    errorBE = compute_errors(uBE, tvec, problem, mode="l2")

    print("\n")
    logger.info("Solving with IMEX-RB")
    uIMEX, *_, iters, _tIMEX = cpu_time(imexrb, problem, u0, [t0, T], Nt,
                                        epsilon, N, maxsubiter)
    errorIMEX = compute_errors(uIMEX, tvec, problem,  mode="l2")

    # Display results in table
    rows = [
            ["Backward Euler", errorBE, _tBE, None],
            [f"IMEX-RB (N={N}, M={maxsubiter}, ε={epsilon:.1e})",
            errorIMEX,
            _tIMEX,
            np.mean(iters)]
        ]

    headers = ["Method", "Rel. Error", "Time (s)", "Avg. Inner Its."]
    table_str = tabulate(
        rows,
        headers=headers,
        tablefmt="github",
        floatfmt=("", ".3e", ".4f", ".2f"),
        missingval=""
    )

    logger.info("\n" + table_str)

    return


if __name__ == "__main__":
    main()
