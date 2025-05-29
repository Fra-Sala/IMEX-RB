import os
import scipy
import numpy as np
import time

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def compute_steps_stability_FE(problem, tspan, factor=1.0):
    """
    Compute minimum number of timesteps to make forward Euler (FE)
    scheme absolutely stable.
    """

    eigvals, _ = scipy.sparse.linalg.eigs(problem.A, k=1, which="LM")
    max_eig = abs(eigvals[0])
    dtFE = factor*2 / max_eig
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


def __custom_slice(y, axis, pos):
    slices = [slice(None)] * y.ndim
    slices[axis] = slice(pos, None) if pos >= 0 else slice(None, pos)
    return y[tuple(slices)]


def integrate_1D(y, x, method='midpoint', axis=0):
    """
    Integrate a 1D array y with respect to x using the specified method.
    """

    if method == 'simpson':
        return scipy.integrate.simpson(y, x, axis=axis)
    elif method == 'midpoint':
        dx = x[1] - x[0]
        return np.sum((__custom_slice(y, axis, 1) + __custom_slice(y, axis, -1)) / 2 * dx,
                      axis=axis)
    else:
        raise ValueError(f"Unknown integration method: '{method}'")


def compute_error_energy(errors_all):
    """
    Compute the energy of errors by summing the squared values along
    the spatial dimension.
    Given input of shape e.g. (soldim, Nx*Ny, Nt), returns
    array of shape (soldim, Nt).
    """
    # Compute squared values
    squared_errors = errors_all**2
    # Sum along the spatial dimension (axis=1)
    error_energy = np.sum(squared_errors, axis=1)

    return error_energy


def cond_sparse(A):
    """
    Compute the condition number in spectral norm of a sparse matrix A
    """

    sigma_max = scipy.sparse.linalg.svds(A, k=1, which='LM',
                                         return_singular_vectors=False)[0]
    sigma_min = scipy.sparse.linalg.svds(A, k=1, which='SM',
                                         return_singular_vectors=False)[0]

    return sigma_max / sigma_min


def get_linear_solver(solver="direct", prec=None):
    """
    Get the linear solver function based on the specified solver type.
    """

    if solver == 'gmres':
        linear_solver = (lambda A, b: scipy.sparse.linalg.gmres(A, b,
                                                                rtol=1e-6, M=prec)[0])
    elif solver == 'direct-sparse':
        linear_solver = scipy.sparse.linalg.spsolve
    elif solver == 'direct':
        linear_solver = scipy.linalg.solve
    else:
        raise ValueError(f"Unknown solver choice: '{solver}'")

    return linear_solver


def create_test_directory(path, testname, n=None):
    """
    Create a test directory for storing results.
    """

    if n is not None and type(n) is int and n > 0:
        test_dir = os.path.join(path, f"{testname}{n}")
    else:
        cnt = 1
        _test_dir = (lambda _cnt: os.path.join(path,
                                               f"{testname}{_cnt}"))
        while os.path.isdir(_test_dir(cnt)):
            cnt += 1
        test_dir = _test_dir(cnt)

    logger.debug(f"Creating test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)

    return test_dir
