from scipy import sparse
from scipy.sparse.linalg import gmres
from scipy.linalg import solve, norm


def newton(problem, Dindx, F, J, x0, tol=1e-8, maxiter=100,
           solverchoice='gmres', option='qNewton'):
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
    solverchoice : {'gmres', 'directsparse', 'dense'}, optional
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
    # Select solver
    if solverchoice == 'gmres':
        linear_solver = (lambda A, b: gmres(A, b)[0])
    elif solverchoice == 'directsparse':
        linear_solver = sparse.linalg.spsolve
    elif solverchoice == 'dense':
        linear_solver = solve
    else:
        raise ValueError(f"Unknown solver choice '{solverchoice}'")

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
        dx_norm = norm(dx)
        if dx_norm < tol:
            info.update({'converged': True,
                         'iterations': i,
                         'final_norm': dx_norm})
            print(f"Newton converged in {i} iters, ||dx|| = {dx_norm:.2e}")
            break
    else:
        # did not break
        info.update({'converged': False,
                     'iterations': maxiter,
                     'final_norm': dx_norm})
        print(f"Newton did not converge in {maxiter} iters, "
              f"final ||dx|| = {dx_norm:.2e}")

    return x, info
