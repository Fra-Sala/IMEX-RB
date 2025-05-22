import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, cond_sparse, \
                        create_test_directory
from utils.errors import compute_errors
from config import Nx, Ny, Lx, Ly, mu, t0, T, Nt, N, maxsubiter
import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """Main file for IMEX-RB applied to Burgers2D"""

    # Setup problem using config.py settings
    problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)
    u0 = problem.initial_condition()
    epsilon = 1.0 / cond_sparse(problem.jacobian(t0, u0))  # epsilon guess
    logger.debug(f"Considering epsilon = {epsilon:.4e}")

    print("\n")
    tvec = np.linspace(t0, T, Nt + 1)

    logger.info("Solving with Backward Euler")
    uBE, *_, _tBE = cpu_time(backward_euler, problem, u0, [t0, T], Nt,
                             solver="direct-sparse")
    errorBE = compute_errors(uBE, tvec, problem, mode="l2")
    # Get error string for all components of velocity
    error_strBE = ", ".join(f"comp{i+1}={err:.4e}"
                            for i, err in enumerate(errorBE))
    logger.info(f"Backward Euler performances:\n"
                f"Relative error: [{error_strBE}];\n"
                f"Computational time: {_tBE:.4f} s.")

    print("\n")
    logger.info("Solving with IMEX-RB")
    uIMEX, *_, iters, _tIMEX = cpu_time(imexrb, problem, u0,
                                        [t0, T], Nt, epsilon,
                                        N, maxsubiter)
    errorIMEX = compute_errors(uIMEX, tvec, problem,  mode="l2")
    error_strIMEX = ", ".join(f"comp{i+1}={err:.4e}"
                              for i, err in enumerate(errorIMEX))
    logger.info(f"IMEX-RB performances (with N={N}, M={maxsubiter}, "
                f"epsilon={epsilon:.4e}):\n"
                f"Relative error: [{error_strIMEX}];\n"
                f"Computational time: {_tIMEX:.4f} s;\n"
                f"Average subiterations number: {np.mean(iters):.4f}.")

    return


if __name__ == "__main__":
    main()
