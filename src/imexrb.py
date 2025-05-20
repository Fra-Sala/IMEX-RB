import numpy as np
import scipy
import time
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
    Didx = problem.dirichlet_idx
    # Setup empty reduced basis
    V = []
    R = []
    subitervec = []
    stability_fails = 0

    for n in range(Nt):
        # Define u_n depending on memory
        uold = u[:, n] if full_u else u_n
        uL = problem.lift_vals(tvec[n + 1])
        # Update subspace with new solution
        V, R, R_update = set_basis(V, R, n, uold, maxsize)
        # Assemble reduced jacobian for quasi-Newton
        JQN = problem.jacobian(tvec[n + 1], uold)
        redjac = V.T @ JQN @ V
        k = 0
        eval_point = uold.copy()

        for k in range(maxsubiter):
            unp1 = np.zeros(np.shape(uold))

            def redF(x):
                """Find RB coefficients x"""
                return x - dt * V.T @ problem.rhs(tvec[n + 1],
                                                  V @ x + uold)
            # Define reduced Jacobian
            redJF = np.identity(V.shape[1]) - dt * redjac
            # Solve for homogeneous reduced solution
            ured, *_ = newton(redF, redJF, np.zeros((V.shape[1]),),
                              solverchoice="dense", option='qNewton')
            # Compute evaluation point for explicit step
            eval_point = V @ ured + uold
            # Enforce BCs (not needed if V is nonhomogeneous)
            # eval_point[Didx] = uL[Didx]
            unp1 = uold + dt * problem.rhs(tvec[n + 1], eval_point)
            # Enforce BCs
            unp1[Didx] = uL[Didx]

            if is_in_subspace(unp1, V, epsilon):
                subitervec.append(k)
                break

            V, R_update = scipy.linalg.qr_insert(
                V, R_update, unp1, V.shape[1], which='col')
            # Update reduced Jacobian
            v_new = V[:, -1]
            V_old = V[:, :-1]
            block12 = V_old.T @ (JQN @ v_new)
            block21 = V_old.T @ (JQN.T @ v_new)
            entry22 = np.array(v_new.T @ (JQN @ v_new))
            # Update reduced Jacobian
            redjac = np.block([
                [redjac, block12[..., np.newaxis]],
                [block21[np.newaxis, ...], entry22]
            ])
            # for refrence: MATLAB equivlaent
            # V = updatebasis(V, utilde);
            # redJ = [redJ, V(:,1:end-1).'*(J_QN*V(:,end));
            # (V(:,1:end-1).'*(J_QN.'*V(:,end))).',   V(:,end).'*(J_QN*V(:,end))];

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


def set_basis(V, R, step, un, maxsize):
    """
    Setup basis for RB step.
    """
    if step == 0 or maxsize == 1:
        # If first timestep, setup initial V
        # or if dim(\mathcal{V}_n) = 1
        V, R = scipy.linalg.qr(un[..., np.newaxis], mode='economic')
    else:
        # Get rid of oldest solution if needed
        if step >= maxsize:
            # We have the QR of U. Obtain new QR of U without the first col
            V, R = \
                scipy.linalg.qr_delete(V, R,
                                       0, 1, which='col',
                                       overwrite_qr=True)
        V, R = \
            scipy.linalg.qr_insert(V, R,
                                   un,
                                   np.shape(V)[1], which='col')

    # Create copy of R for subiterations update
    R_update = R.copy()

    return V, R, R_update
