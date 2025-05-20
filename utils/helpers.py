import os
import scipy
import numpy as np
import time

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def compute_steps_stability_FE(problem, tspan, factor=0.95):
    """
    Compute minimum number of timesteps to make forward Euler (FE)
    scheme absolutely stable.
    """

    eigvals, _ = scipy.sparse.linalg.eigs(problem.A, k=1, which="LM")
    max_eig = abs(eigvals[0])
    dtFE = factor*2 / max_eig  # use 90% of the stability limit
    Nt_FE = int(np.ceil((tspan[1]-tspan[0]) / dtFE)) + 1

    return Nt_FE


def cpu_time(func, *args, **kwargs):
    """
    Wrapper to measure CPU execution time of a function.
    """

    start_time = time.process_time()
    result = func(*args, **kwargs)
    end_time = time.process_time()
    _cpu_time = end_time - start_time

    return *result, _cpu_time


def integrate_1D(y, x):
    return scipy.integrate.simpson(y, x)


def cond_sparse(A):
    """
    Compute the condition number in spectral norm of a sparse matrix A
    """

    sigma_max = scipy.sparse.linalg.svds(A, k=1, which='LM',
                                         return_singular_vectors=False)[0]
    sigma_min = scipy.sparse.linalg.svds(A, k=1, which='SM',
                                         return_singular_vectors=False)[0]

    return sigma_max / sigma_min


def create_test_directory(path, n=None):
    """
    Create a test directory for storing results.
    """

    if n is not None and type(n) is int and n > 0:
        test_dir = os.path.join(path, "NvsNt", f"Test{n}")
    else:
        cnt = 1
        _test_dir = lambda _cnt: os.path.join(path, "NvsNt", f"Test{_cnt}")
        while os.path.isdir(_test_dir(cnt)):
            cnt += 1
        test_dir = _test_dir(cnt)

    logger.debug(f"Creating test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)

    return test_dir
