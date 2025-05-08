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
        if not check_presence(u[:, n], V, epsilon):
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
            ured = scipy.linalg.solve((np.identity(np.shape(V)[1]) - dt*Ared),
                                      bred, assume_a='general')
            # Full order step
            eval_point = V @ ured + u[:, n] - V @ (V.T @ u[:, n])
            unew = u[:, n] + dt * (problem.A @ eval_point + sourcetn)
            unew = problem.enforce_bcs(unew, tvec[n+1])
            if check_presence(unew, V, epsilon):
                break
            else:
                # Add unew (subiterate) to the basis
                V, R = scipy.linalg.qr_insert(V, R, unew, np.shape(V)[1],
                                              which='col')
            k += 1
        subitervec.append(k+1)
        # Use qr_delete to get rid of possible subiters. keep only up to N cols
        if np.shape(V)[1] > maxsize:
            V, R = scipy.linalg.qr_delete(V, R, maxsize-1,
                                          np.shape(V)[1] - maxsize,
                                          which='col',
                                          overwrite_qr=True)
        u[:, n+1] = unew
    return u, tvec, subitervec, time.time() - start


def check_presence(vec, basis, epsilon):
    """
    Check if || (I-VV^T)*vec || < epsilon*||vec||
    """
    residual = np.linalg.norm(vec - basis @ ((basis.T) @ vec)) / \
        np.linalg.norm(vec)
    # print(f"Current residual {residual}\n")
    return residual < epsilon