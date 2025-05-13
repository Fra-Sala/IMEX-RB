import numpy as np
import scipy
import time
from scipy import sparse


def imexrb(problem,
           u0,
           tspan,
           Nt,
           epsilon,
           maxsize,
           maxsubiter,
           contain_un=False):
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
    # Compute once for all system matrix I - dt A
    M = sparse.identity(problem.Nh, format='csr') - dt * problem.A
    # Get rid of Dirichlet rows and columns
    Mmod = problem.modify_system_matrix(M)
    # Retrieve non-Dirichlet indices
    Inodes = problem.non_dirichlet_indices
    # Setup reduced basis
    V, R = scipy.linalg.qr(u0[Inodes, np.newaxis], mode='economic')
    subitervec = []

    for n in range(Nt):
        # Define u(t_n) depending on memory
        uold = u[:, n] if full_u else u_n
        # Update subspace if needed
        if not is_in_subspace(uold[Inodes],
                              V, epsilon):
            if V.shape[1] >= maxsize:
                V, R = scipy.linalg.qr_delete(
                    V, R, 0, 1, which='col', overwrite_qr=True)
            V, R = scipy.linalg.qr_insert(
                V, R,
                uold[Inodes],
                V.shape[1], which='col')
        # Compute once for all source terms
        sourcetnp1 = problem.source_term(tvec[n + 1])
        sourcetn = problem.source_term(tvec[n])
        # Build BE rhs with lifting
        bmod, uL = problem.apply_lifting(M, dt*sourcetnp1, tvec[n + 1])
        # Compute forcing term for explicit step
        bEX, _ = problem.apply_lifting(-dt*problem.A, dt*sourcetn, tvec[n + 1])
        Amod = problem.modify_system_matrix(problem.A)
        k = 0
        while k < maxsubiter:
            unew = np.zeros(np.shape(uold))
            Mred = V.T @ Mmod @ V
            bred = V.T @ (
                uold[Inodes] + bmod)

            # Solve for homogeneous reduced solution
            ured = scipy.linalg.solve(
                Mred,  bred,
                assume_a='general')
            # Compute evaluation point for explicit step
            eval_point = V @ ured + uold[Inodes] - V @ (V.T @ uold[Inodes])
            unew[Inodes] = uold[Inodes] + dt * Amod * eval_point + bEX
            # Enforce BCs
            unew += uL

            if is_in_subspace(unew[Inodes], V, epsilon):
                break

            V, R = scipy.linalg.qr_insert(
                V, R, unew[Inodes], V.shape[1], which='col')
            k += 1

        subitervec.append(k)

        # Trim basis
        if V.shape[1] > maxsize:
            V, R = scipy.linalg.qr_delete(
                V, R, maxsize - 1,
                V.shape[1] - maxsize,
                which='col', overwrite_qr=True)

        # Store new solution
        if full_u:
            u[:, n + 1] = unew
        else:
            u_n = unew

    elapsed = time.time() - start
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
