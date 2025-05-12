import numpy as np
import scipy
import time


def imexrb(problem, u0, tspan, Nt, epsilon, maxsize, maxsubiter,
           contain_un=False):
    start = time.time()
    t0, tf = tspan
    tvec, dt = np.linspace(t0, tf, Nt+1, retstep=True)
    u = np.empty((np.shape(u0)[0], Nt+1))
    u[:, 0] = u0
    # Setup basis
    V, R = scipy.linalg.qr(u0[..., np.newaxis], mode='economic')
    subitervec = []
    for n in range(Nt):
        if not is_in_subspace(u[:, n], V, epsilon):
            # If u_n is not inside V, add it
            if n >= maxsize:
                # Get rid of oldest solution
                V, R = scipy.linalg.qr_delete(V, R,
                                              0, 1, which='col',
                                              overwrite_qr=True)
            # Add u_n to the subspace
            V, R = scipy.linalg.qr_insert(V, R, u[:, n], np.shape(V)[1],
                                          which='col')
        k = 0
        # Precompute vectors once for all
        sourcetnp1 = problem.source_term(tvec[n+1])
        sourcetn = problem.source_term(tvec[n])
        while k < maxsubiter:
            Ared = V.T @ problem.A @ V
            bred = V.T @ (u[:, n] + dt*sourcetnp1)
            # Reduced step
            ured = scipy.linalg.solve((scipy.sparse.identity(np.shape(V)[1],
                                       format='csr') - dt*Ared),
                                      bred, assume_a='general')
            # Full order step
            eval_point = V @ ured + u[:, n] - V @ (V.T @ u[:, n])
            unew = u[:, n] + dt * (problem.A @ eval_point + sourcetn)
            unew = problem.enforce_bcs(unew, tvec[n+1])
            if is_in_subspace(unew, V, epsilon):
                break
            else:
                # Add unew (subiterate) to the basis
                V, R = scipy.linalg.qr_insert(V, R, unew, np.shape(V)[1],
                                              which='col')
            k += 1
        subitervec.append(k+1)
        # qr_delete to get rid of possible subiters
        # Keep only up to maxsize cols
        if np.shape(V)[1] > maxsize:
            V, R = scipy.linalg.qr_delete(V, R, maxsize-1,
                                          np.shape(V)[1] - maxsize,
                                          which='col',
                                          overwrite_qr=True)
        u[:, n+1] = unew
    return u, tvec, subitervec, time.time() - start


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
