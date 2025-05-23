import os
import sys
import numpy as np
import cProfile
import pstats
# TO DO: fix import hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import Burgers2D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import cpu_time, integrate_1D, cond_sparse, \
    create_test_directory
from utils.errors import compute_errors

from config import Nx, Ny, Lx, Ly, mu, t0, T, Nt, N, maxsubiter, \
    results_dir

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """We measure CPU times of IMEX-RB, BE and FE applied to the nonlinear
    2D Burgers equation, varying the size of the problem Nh"""

    # Define profiling directory
    testname = "profiling"
    test_dir = create_test_directory(os.path.join(results_dir, "Burgers2D"),
                                     testname)
    epsilon = 1e-4
    logger.debug(f"Considering epsilon = {epsilon}")

    # Setup problem
    problem = Burgers2D(Nx, Ny, Lx, Ly, mu=mu)
    u0 = problem.initial_condition()

    logger.info(f"Solving for Nh={problem.Nh}")

    # Profile Backward Euler
    logger.info("Profiling Backward Euler")
    be_profile = cProfile.Profile()
    be_profile.enable()
    uBE, *_ = backward_euler(problem, u0, [t0, T], Nt,
                             solver="direct-sparse")
    be_profile.disable()
    be_stats = pstats.Stats(be_profile)
    be_stats.dump_stats(os.path.join(test_dir, "backward_euler_profile.prof"))

    # Profile IMEX-RB
    logger.info(f"Profiling IMEX-RB with N={N}")
    imexrb_profile = cProfile.Profile()
    imexrb_profile.enable()
    uIMEX, *_, iters = imexrb(problem, u0, [t0, T], Nt, epsilon, N,
                              maxsubiter)
    imexrb_profile.disable()
    imexrb_stats = pstats.Stats(imexrb_profile)
    imexrb_stats.dump_stats(os.path.join(test_dir, "imexrb_profile.prof"))

    # Save additional information
    np.savez(os.path.join(test_dir, "profiling_info.npz"),
             problem_size=problem.Nh,
             Nt=Nt,
             N=N,
             avg_subiters=np.mean(iters))

    logger.info(f"Profiling results saved to {test_dir}")

    return


if __name__ == "__main__":
    main()