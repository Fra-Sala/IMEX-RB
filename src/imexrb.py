import numpy as np
import scipy
import time
from scipy import sparse
from newton import newton


def imexrb(problem,
           u0,
           tspan,
           Nt,
           epsilon,
           maxsize,
           maxsubiter):
    """
    IMEX-RB time integration with memory fallback for large u allocation.

    If full history allocation fails, only current and next solution are kept.
    """
    start = time.time()
    t0, tf = tspan
    tvec, dt = np.linspace(t0, tf, Nt + 1, retstep=True)
    try:
        u = np.empty((u0.shape[0], Nt + 1))
        u[:, 0] = u0
        full_u = True
    except MemoryError:
        # Fallback: keep only current and next solution
        u_n = u0.copy()
        full_u = False

    # Retrieve non-Dirichlet indices
    Dindx = problem.dirichlet_indices
    # Setup reduced basis
    V, R = scipy.linalg.qr(u0[..., np.newaxis], mode='economic')
    subitervec = []
    stability_fails = 0

    for n in range(Nt):
        # Define u(t_n) depending on memory
        uold = u[:, n] if full_u else u_n
        uold0 = uold.copy()
        uold0[Dindx] = 0.0
        uL = problem.compute_bcs(tvec[n + 1])
        # Update subspace with new solution
        if n != 0:
            if V.shape[1] >= maxsize:
                V, R = scipy.linalg.qr_delete(
                    V, R, 0, 1, which='col', overwrite_qr=True)
            V, R = scipy.linalg.qr_insert(
                V, R,
                uold,
                V.shape[1], which='col')

        # Assemble reduced jacobian
        redjac = V.T @ problem.jacobian(tvec[n + 1], uold) @ V
        k = 0
        for k in range(maxsubiter):
            unp1 = np.zeros(np.shape(uold))

            def redF(x):
                """Find RB coefficients x"""
                return x - dt * V.T @ problem.rhs(tvec[n + 1],
                                                  V @ x + uold0 + uL)
            # Define reduced Jacobian
            redJF = np.identity(V.shape[1]) - dt * redjac
            # Solve for homogeneous reduced solution
            ured, *_ = newton(redF, redJF, V.T @ uold0,
                              solverchoice="dense", option='qNewton')
            # Compute evaluation point for explicit step
            eval_point = V @ ured + uold0 + uL
            unp1 = uold + dt * problem.rhs(tvec[n + 1], eval_point)
            # Enforce BCs
            unp1[Dindx] += uL[Dindx]

            if is_in_subspace(unp1, V, epsilon):
                subitervec.append(k)
                break

            V, R = scipy.linalg.qr_insert(
                V, R, unp1, V.shape[1], which='col')
            # Update reduced Jacobian
            
        else:
            stability_fails += 1
            subitervec.append(maxsubiter)

        # Trim basis
        if V.shape[1] > maxsize:
            V, R = scipy.linalg.qr_delete(
                V, R, maxsize - 1,
                V.shape[1] - maxsize,
                which='col', overwrite_qr=True)

        # Store new solution
        if full_u:
            u[:, n + 1] = unp1
        else:
            u_n = unp1

    elapsed = time.time() - start
    # Print a message to warn for absolute stability not met
    print(f"Stability condition NOT met (times/total): {stability_fails}/{Nt}")
    if full_u:
        return u, tvec, subitervec, elapsed
    # Only last solution available
    return u_n, tvec, subitervec, elapsed


def is_in_subspace(vec, basis, epsilon):
    """
    Determines if a given vector lies approximately within a subspace 
    spanned by a given basis, within a specified tolerance.

    Parameters:
    -----------
    vec : numpy.ndarray
        The vector to be checked.
    basis : numpy.ndarray
        The matrix whose columns form an orthonormal basis for the subspace.
    epsilon : float
        The tolerance for determining if the vector is in the subspace.

    Returns:
    --------
    bool
        True if the vector is approximately in the subspace, False otherwise.
    """
    residual = np.linalg.norm(vec - basis @ ((basis.T) @ vec)) / \
        np.linalg.norm(vec)
    # print(f"Current residual {residual}\n")
    return residual < epsilon
