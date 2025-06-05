import os
import sys
import numpy as np
import cProfile
import pstats
# TO DO: fix import hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from src.problemsPDE import AdvDiff3D
from src.euler import backward_euler
from src.imexrb import imexrb
from utils.helpers import create_test_directory
from examples.advDiff3D.config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """We profile IMEX-RB, BE looking for bottlenecks"""

    # Define profiling directory
    Nx = 76
    Ny = 76
    Nz = 76
    testname = "profiling"
    project_path = os.getcwd()
    profile_dir = os.path.join(project_path, "profiling")

    epsilon = 0.003283019192460771
    Nt = 128
    logger.debug(f"Considering epsilon = {epsilon}")
    logger.debug(f"Solving for N={N}")
    logger.debug(f"Solving for M={maxsubiter}")
    # Setup problem
    problem = AdvDiff3D(Nx, Ny, Nz, Lx, Ly, Lz, mu=mu,
                        sigma=sigma, vx=vx, vy=vy, vz=vz, center=center)

    test_dir = create_test_directory(os.path.join(profile_dir, problem.name),
                                     testname)
    u0 = problem.initial_condition()
    logger.debug(f"Solving for Nh={problem.Nh}")
    logger.debug(f"Solving for Nt={Nt}")

    # Profile Backward Euler
    logger.info("Profiling Backward Euler")
    be_profile = cProfile.Profile()
    be_profile.enable()
    uBE, *_ = backward_euler(problem, u0, [t0, T], Nt,
                             solver='gmres')
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

    logger.info(f"IMEX-RB performed subiters (last run): "
                f"avg={np.mean(iters)}, max={np.max(iters)}, "
                f"tot={np.sum(iters)}")

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