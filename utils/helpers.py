import os
import scipy
import numpy as np
import time

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def compute_steps_stability_FE(problem, tspan, factor=0.99,
                               tol=1e-8, path=None):
    """
    Compute the number of time steps required for stability of the
    Forward Euler (FE) method. It optionally
    saves and loads the computed number of time steps to/from a file.

    Parameters:
    ----------
        problem (object): An object containing the system matrix `A`.
        tspan (tuple): A tuple specifying the time interval (start, end).
        factor (float, optional): A safety factor to scale the
            maximum \\Delta t.
        tol (float, optional): Tolerance for the eigenvalue computation.
            Default is 1e-8.
        path (str, optional): Path to save or load the computed number of
            time steps. If `None`, no file operations are performed.

    Returns:
    -------
        int: The number of time steps required for stability of the
        Forward Euler method.
    """

    if path is not None:
        steps_file = os.path.join(path, "steps_FE.npz")
        if os.path.exists(steps_file):
            data = dict(np.load(steps_file, allow_pickle=True))
            return data["Nt_FE"]
        # Create the directory if it does not exist
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    eigvals, _ = scipy.sparse.linalg.eigs(problem.A, k=1, which="LM", tol=tol)
    max_eig = abs(eigvals[0])
    dtFE = factor * 2 / max_eig
    Nt_FE = int(np.ceil((tspan[1] - tspan[0]) / dtFE)) + 1

    if path is not None:
        data = {"Nt_FE": Nt_FE, "tol": tol}
        np.savez(steps_file, **data)
        return data["Nt_FE"]
    else:
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
        return np.sum((__custom_slice(y, axis, 1) +
                       __custom_slice(y, axis, -1)) / 2 * dx,
                      axis=axis)
    else:
        raise ValueError(f"Unknown integration method: '{method}'")


def cond_sparse(A, tol=1e-8, path=None):
    """
    Compute the condition number of a sparse matrix using its singular values.

    This function calculates the condition number of a sparse matrix `A` as
    the ratio of its largest singular value to its smallest singular value.
    Optionally, the result can be cached to a file for reuse.

    Parameters:
    ----------
        A (scipy.sparse.spmatrix): The sparse matrix for which the condition
        number is to be computed.
        tol (float, optional): Tolerance for the singular value decomposition.
            Default is 1e-8.
        path (str, optional): Directory path to save or load the cached
            condition number. If provided, the function will check for a
            cached result in `cond.npz` within the specified directory.
            If the file does not exist, it will compute the condition number
            and save it to this file.

    Returns:
    --------
        float: The condition number of the matrix `A`.
    """

    if path is not None:
        cond_file = os.path.join(path, "cond.npz")
        if os.path.exists(cond_file):
            data = dict(np.load(cond_file, allow_pickle=True))
            return data["cond"]
        # Create the directory if it does not exist
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    sigma_max = scipy.sparse.linalg.svds(A, k=1, which='LM', tol=tol,
                                         return_singular_vectors=False)[0]
    sigma_min = scipy.sparse.linalg.svds(A, k=1, which='SM', tol=tol,
                                         return_singular_vectors=False)[0]
    cond_value = sigma_max / sigma_min

    if path is not None:
        data = {"cond": cond_value, "tol": tol}
        np.savez(cond_file, **data)
        return data["cond"]
    else:
        return cond_value


def get_linear_solver(solver="direct", prec=None):
    """
    Get the linear solver function based on the specified solver type.
    """

    if solver == 'gmres':
        linear_solver = (lambda A, b:
                         scipy.sparse.linalg.gmres(A, b,
                                                   rtol=1e-10, M=prec)[0])

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
