
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
import numpy as np
import timeit

from src.problemsPDE import AdvDiff2D
from src.euler import backward_euler, forward_euler
from src.imexrb import imexrb
from utils.helpers import integrate_1D, cond_sparse, create_test_directory, \
    compute_steps_stability_FE
from utils.errors import compute_errors

from numerical_tests.advDiff2D.config import *

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def main():
    """
    In this test, we evaluate the performances of IMEX-RB, compared
    to those of backward Euler, considering different timestep values.
    We investigate the 2D advection diffusion problem
    In particular, we study:
    Convergence, average inner iter, and CPU time vs \\Delta t.
    -- Multiple curves are produced varying N. --
    We will then also plot the iterations over time for a given
    \\Delta t.
    """
    # Define test parameters
    Nx_values = Ni_values
    N_values = [1, 5, 10, 15, 20]  # minimal dimension of the reduced basis
    Nt_values = [2 ** n for n in range(4, 11)]  # range of Nt values

    n_solves = 10  # number of solver calls to estimate computational times
    # Initialise variables to track method performances
    errors_l2 = {
        "IMEX-RB": np.empty((len(Nt_values), len(N_values), len(Nx_values))),
        "BE": np.empty((len(Nt_values), len(Nx_values))),
        "FE": np.empty((len(Nt_values), len(Nx_values)))
    }
    errors_all = {
        "IMEX-RB": np.empty(
            (len(Nt_values), len(N_values), Nt_values[-1], len(Nx_values))
        ),
        "BE": np.empty((len(Nt_values), Nt_values[-1], len(Nx_values))),
        "FE": np.empty((len(Nt_values), Nt_values[-1], len(Nx_values)))
    }
    times = {
        "IMEX-RB": np.zeros((len(Nt_values), len(N_values), len(Nx_values))),
        "BE": np.zeros((len(Nt_values), len(Nx_values))),
        "FE": np.empty((len(Nt_values), len(Nx_values)))
    }
    subiters = {
        "IMEX-RB": np.empty(
            (len(Nt_values), len(N_values), Nt_values[-1], len(Nx_values))
        ),
        "BE": None,
        "FE": None
    }
    Nt_FEs = np.empty(len(Nx_values))

    for idx_Nx, Nx in enumerate(Nx_values):
        print("\n")
        logger.info(f"Solving for Nx={Nx}")
        # Setup problem for the current Nx
        Ny = Nx
        problem = AdvDiff2D(Nx, Ny, Lx, Ly, mu=mu, sigma=sigma,
                            vx=vx, vy=vy, center=center)
        if idx_Nx == 0:
            # Define test directory and initial condition
            testname = "NvsNt"
            test_dir = create_test_directory(os.path.join(results_dir, problem.name),
                                             testname)
        u0 = problem.initial_condition()
        epsilon = 1.0 / cond_sparse(
            problem.A,
            tol=tol_cond_NtFE,
            path=os.path.join(results_dir, problem.name,
                            f"params_Nh_{problem.Nh}")
        )
        logger.debug(f"Considering epsilon = {epsilon:.4e}")

        Nt_FEs[idx_Nx] = compute_steps_stability_FE(
            problem, [t0, T], tol=tol_cond_NtFE,
            path=os.path.join(results_dir, problem.name,
                              f"params_Nh_{problem.Nh}")
        )
        for cnt_Nt, _Nt in enumerate(Nt_values):
            print("\n")
            logger.info(f"Solving for Nt={_Nt}")
            tvec = np.linspace(t0, T, _Nt + 1)

            logger.info("Solving with Backward Euler (BE)")
            uBE, *_ = backward_euler(problem, u0, [t0, T], _Nt,
                                     **sparse_solver)

            if n_solves > 0:
                f_BE = (lambda: backward_euler(problem, u0, [t0, T], _Nt,
                                              **sparse_solver))
                timer = timeit.Timer(f_BE)
                _t = timer.repeat(number=1, repeat=n_solves)
                times["BE"][cnt_Nt, idx_Nx] += np.mean(_t)

            errors_all["BE"][cnt_Nt, :_Nt, idx_Nx] = compute_errors(
                uBE, tvec, problem, mode="all"
            )
            errors_l2["BE"][cnt_Nt, idx_Nx] = integrate_1D(
                errors_all["BE"][cnt_Nt, :_Nt, idx_Nx], tvec[1:]
            )

            logger.info("Solving with Forward Euler (FE)")
            uFE, *_ = forward_euler(problem, u0, [t0, T], _Nt)

            if n_solves > 0:
                f_FE = (lambda: forward_euler(problem, u0, [t0, T], _Nt))
                timer = timeit.Timer(f_FE)
                _t = timer.repeat(number=1, repeat=n_solves)
                times["FE"][cnt_Nt, idx_Nx] += np.mean(_t)

            errors_all["FE"][cnt_Nt, :_Nt, idx_Nx] = compute_errors(
                uFE, tvec, problem, mode="all"
            )
            errors_l2["FE"][cnt_Nt, idx_Nx] = integrate_1D(
                errors_all["FE"][cnt_Nt, :_Nt, idx_Nx], tvec[1:]
            )

            logger.info("Solving with IMEX-RB")
            for cnt_N, N in enumerate(N_values):
                logger.info(f"Solving for N={N}")

                uIMEX, _, iters = imexrb(
                    problem, u0, [t0, T], _Nt, epsilon, N, maxsubiter
                )

                if n_solves > 0:
                    f_IMEX = (lambda: imexrb(problem, u0, [t0, T], _Nt,
                                             epsilon, N, maxsubiter))
                    timer = timeit.Timer(f_IMEX)
                    _t = timer.repeat(number=1, repeat=n_solves)
                    times["IMEX-RB"][cnt_Nt, cnt_N, idx_Nx] += np.mean(_t)

                # Store subiterates
                subiters["IMEX-RB"][cnt_Nt, cnt_N, :_Nt, idx_Nx] = iters

                # Compute errors
                errors_all["IMEX-RB"][cnt_Nt, cnt_N, :_Nt, idx_Nx] = \
                    compute_errors(uIMEX, tvec, problem, mode="all")
                errors_l2["IMEX-RB"][cnt_Nt, cnt_N, idx_Nx] = integrate_1D(
                    errors_all["IMEX-RB"][cnt_Nt, cnt_N, :_Nt, idx_Nx], tvec[1:]
                )

    np.savez(
        os.path.join(test_dir, "results.npz"),
        Nt_FE=Nt_FEs,
        errors_l2=errors_l2,
        errors_all=errors_all,
        times=times,
        subiters=subiters,
        N_values=N_values,
        Nt_values=Nt_values,
        Nx_values=Nx_values
    )

    return


if __name__ == "__main__":
    main()
