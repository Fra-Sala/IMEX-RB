import os
import scipy.linalg
from utils.helpers import get_linear_solver

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def newton(F, J, x0, tol=1e-8, maxiter=100,
           solver='gmres', option='qNewton',
           is_linear=False, prec=None):
    """
    Perform Newton's method to solve a nonlinear system F(x) = 0.

    Parameters
    ----------
    F : callable
        Residual function, F(x) -> r.
    J : array_like or callable
        Jacobian matrix or function J(x) -> J.
    x0 : ndarray
        Initial guess.
    tol : float, optional
        Convergence tolerance on correction norm.
    maxiter : int, optional
        Maximum number of Newton iterations.
    solver : {'gmres', 'direct-sparse', 'direct'}, optional
        Linear solver for J dx = r.
    option : {'qNewton', 'Newton'}, optional
        'qNewton' uses fixed J, 'Newton' updates J each iteration.

    Returns
    -------
    x : ndarray
        Approximate solution.
    info : dict
        Dictionary with convergence info.
    """

    linear_solver = get_linear_solver(solver=solver, prec=prec)

    x = x0.copy()
    info = {'converged': False, 'iterations': 0, 'final_norm': None}

    # Optionally compute constant Jacobian
    if option == 'qNewton' and not callable(J):
        J_const = J

    for i in range(1, maxiter + 1):
        # Evaluate residual
        res = F(x)
        jac = J_const if (option == 'qNewton' and not callable(J)) \
            else (J(x))
        dx = linear_solver(jac, res)
        x -= dx

        if is_linear:
            break

        dx_norm = scipy.linalg.norm(dx)
        if dx_norm < tol:
            info.update({'converged': True,
                         'iterations': i,
                         'final_norm': dx_norm})
            break

    else:
        # did not break
        info.update({'converged': False,
                     'iterations': maxiter,
                     'final_norm': dx_norm})

        logger.warning(f"Newton did not converge in {maxiter} iters, "
                       f"final ||dx|| = {dx_norm:.2e}")

    return x, info
