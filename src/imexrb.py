import os
import numpy as np
import scipy.linalg
from src.newton import newton
from functools import partial
import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def imexrb(problem, u0, tspan, Nt, epsilon, maxsize, maxsubiter):
    """
    Implicit-Explicit Reduced Basis (IMEX-RB) time integration method
    for solving systems of ordinary differential equations (ODEs).

    Parameters:
    ----------
        problem : object
            An object representing the problem to be solved.

        u0 : ndarray
            Initial condition vector.
        tspan : tuple
            A tuple (t0, tf) specifying the initial and final times.
        Nt : int
            Number of time steps.
        epsilon : float
            Stability tolerance.
        maxsize : int
            Minimal and initial size of the reduced basis.
        maxsubiter : int
            Maximum number of inner iterations to be performed by
            the algorithm.
    Returns:
    --------
    u : ndarray
        The solution array of shape `(problem.Nh, Nt + 1)` containing the
        solution at all time steps. Returned only if full history allocation
        succeeds.

    un : ndarray
        The solution at the final time step. Returned only if full history
        allocation fails.

    tvec : ndarray
        The time vector of shape `(Nt + 1,)` containing the time points
        corresponding to the solution.

    subitervec : ndarray
        The vector of the number of inner iterations performed at each
        timestep.
    """
    t0, tf = tspan
    tvec, dt = np.linspace(t0, tf, Nt + 1, retstep=True)
    try:
        u = np.empty((u0.shape[0], Nt + 1))
        u[:, 0] = u0
        full_u = True
    except MemoryError:
        # Fallback: keep only current and next solution
        logger.info("Memory issue for IMEX-RB. Saving only un \n")
        u_n = u0.copy()
        full_u = False

    # Retrieve non-Dirichlet (free) indices
    Didx = problem.dirichlet_idx
    free_idx = problem.free_idx
    # Setup empty reduced basis
    V = []
    R = []
    subitervec = []
    stability_fails = 0
    # Allocate array to store new solution
    unp1 = np.empty_like(u0)

    for n in range(Nt):
        # Define u_n depending on memory
        uold = u[:, n] if full_u else u_n
        uL = problem.get_boundary_values(tvec[n + 1])
        # Update subspace with new solution
        V, R, R_update = set_basis(V, R, n, uold[free_idx], maxsize)
        # Assemble reduced jacobian for quasi-Newton
        JQN = problem.jacobian_free(tvec[n + 1], uold)
        redjac = V.T @ (JQN @ V)

        for k in range(maxsubiter):
            # Set new solution to 0
            unp1.fill(0)
            # Define reduced nonlinear problem
            uold_free = uold[free_idx]
            current_t = tvec[n+1]
            rhs = problem.rhs_free
            # Create a 1-arg function for Newton
            redF = partial(redF_full, V=V,
                           uold_free=uold_free,
                           t=current_t, dt=dt,
                           rhs_free=rhs)
            # Define reduced Jacobian
            redJF = np.identity(V.shape[1]) - dt * redjac
            # Solve for homogeneous reduced solution
            ured, *_ = newton(redF, redJF, np.zeros((V.shape[1]),),
                              solver="direct", option='qNewton',
                              tol=1e-3*dt**2)
            # Compute evaluation point for explicit step
            eval_point = V @ ured + uold[free_idx]
            unp1[free_idx] = uold[free_idx] +\
                dt * problem.rhs_free(tvec[n + 1], eval_point)
            # Enforce BCs
            unp1[Didx] = uL[Didx]

            if is_in_subspace(unp1[free_idx], V, epsilon):
                subitervec.append(k)
                break

            V, R_update = scipy.linalg.qr_insert(
                V, R_update, unp1[free_idx], V.shape[1], which='col')

            # Update reduced Jacobian
            v_new = V[:, -1]
            V_old = V[:, :-1]
            block12 = V_old.T @ (JQN @ v_new)
            block21 = (V_old.T @ (JQN.T @ v_new)).T
            entry22 = np.array(v_new.T @ (JQN @ v_new))

            redjac = np.block([
                [redjac, block12[..., np.newaxis]],
                [block21[np.newaxis, ...], entry22]
            ])

        else:
            stability_fails += 1
            subitervec.append(maxsubiter)

        # Trim basis
        if subitervec[-1] > 0:
            V = V[:, :(-subitervec[-1])]

        # Store new solution
        if full_u:
            u[:, n + 1] = unp1
        else:
            u_n = unp1

    logger.debug(f"IMEX-RB: stability condition NOT met (times/total):"
                 f"{stability_fails}/{Nt}")

    if full_u:
        return u, tvec, subitervec

    return u_n, tvec, subitervec


def redF_full(x, V, uold_free, t, dt, rhs_free):
    """
    Reduced nonlinear residual:
    redF(x) = x - dt * V^T rhs_free(t, V * x + uold[free_idx])
    """
    return x - dt * (V.T @ rhs_free(t, V @ x + uold_free))


def is_in_subspace(vec, basis, epsilon):
    """
    Determines if a given vector lies approximately within a subspace
    spanned by a given basis, within the stability tolerance.

    Parameters:
    -----------
    vec : numpy.ndarray
        The vector to be checked (candidate u_n+1).
    basis : numpy.ndarray
        The matrix whose columns form an orthonormal basis for the subspace.
    epsilon : float
        The stability tolerance.

    Returns:
    --------
    bool
        True if the vector is approximately in the subspace, False otherwise.
    """
    residual = \
        np.linalg.norm(vec - basis @ (basis.T @ vec)) / np.linalg.norm(vec)

    return residual < epsilon


def set_basis(V, R, step, un, maxsize):
    """
    Update the basis matrix `V` and the upper triangular matrix `R` using
    QR update, with the option to manage the size of the basis.

    Parameters:
    -----------
    V : ndarray
        The current basis matrix (orthonormal columns).
    R : ndarray
        The current upper triangular matrix from the QR factorization.
    step : int
        The current timestep index.
    un : ndarray
        The vector to be added to the basis.
    maxsize : int
        The maximum allowable size (number of columns) of the basis matrix `V`.

    Returns:
    --------
    V : ndarray
        The updated basis matrix after QR factorization.
    R : ndarray
        The updated upper triangular matrix after QR factorization.
    R_update : ndarray
        A copy of the updated `R` matrix for subiteration updates.

    Notes:
    ------
    - At the first timestep (`step == 0`) or if the maximum size of the basis
      is 1 (`maxsize == 1`), the basis is initialized using the vector `un`.
    - If the maximum size of the basis is exceeded, the oldest column in `V`
      is removed before adding the new vector.
    - If the new vector `un` is already in the span of the current basis,
      a `LinAlgError` is caught, and the basis remains unchanged.
    """

    if step == 0 or maxsize == 1:
        # If first timestep, setup initial V
        # or if dim(\mathcal{V}_n) = 1
        V, R = scipy.linalg.qr(un[..., np.newaxis], mode='economic')
    else:
        # Get rid of oldest solution if needed
        try:
            if step >= maxsize:
                #  t0_del = time.perf_counter()
                # We have the QR of U, we compute QR of U without the first col
                V, R = \
                    scipy.linalg.qr_delete(V, R,
                                           0, 1, which='col',
                                           overwrite_qr=True)

            V, R = \
                scipy.linalg.qr_insert(V, R,
                                       un,
                                       np.shape(V)[1], which='col',
                                       rcond=1e-8)
        except scipy.linalg.LinAlgError:  # u_n is already in the span
            # do nothing, i.e. keep V and R before the try
            logger.debug("LinAlgError caught: keeping previous V and R")

    # Create copy of R for subiterations update
    R_update = R.copy()

    return V, R, R_update
