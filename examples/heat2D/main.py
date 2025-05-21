import os
import numpy as np

from src.problemsPDE import Heat2D
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
    """Main file for IMEX-RB"""

    # Setup problem
    problem = Heat2D(Nx, Ny, Lx, Ly, mu=mu, sigma=sigma, center=center)

    u0 = problem.initial_condition()
    epsilon = 1.0 / cond_sparse(problem.A)  # epsilon for absolute stability condition --> OK since A is symmetric
    logger.debug(f"Considering epsilon = {epsilon:.4e}")

    print("\n")
    tvec = np.linspace(t0, T, Nt + 1)

    logger.info("Solving with Backward Euler")
    uBE, *_, _tBE = cpu_time(backward_euler, problem, u0, [t0, T], Nt, solver="direct-sparse")
    errorBE = compute_errors(uBE, tvec, problem, mode="l2")
    logger.info(f"Backward Euler performances:\n"
                f"Relative error: {errorBE:.4e};\n"
                f"Computational time: {_tBE:.4f} s.")

    print("\n")
    logger.info("Solving with IMEX-RB")
    uIMEX, *_, iters, _tIMEX = cpu_time(imexrb, problem, u0, [t0, T], Nt, epsilon, N, maxsubiter)
    errorIMEX = compute_errors(uIMEX, tvec, problem,  mode="l2")
    logger.info(f"IMEX-RB performances (with N={N}, M={maxsubiter}, epsilon={epsilon:.4e}):\n"
                f"Relative error: {errorIMEX:.4e};\n"
                f"Computational time: {_tIMEX:.4f} s;\n"
                f"Average subiterations number: {np.mean(iters):.4f}.")

    return


if __name__ == "__main__":
    main()
