import numpy as np


def compute_errors(u, tvec, xvec, exact_solution, dim=1, q=2,
                   finaltimeonly=False):
    """
    Compute FD errors over a dim dimensional domain.
    The formula for the error is taken from 
    'Randall J. LeVeque. Finite Difference Methods for Ordinary
    and Partial Differential Equations.'

    """
    
    if finaltimeonly:
        # Consider only final time and final solution
        tvec = tvec[-2:]
        u = u[..., -1]
    
    err_q_x = np.zeros(len(tvec) - 1)

    if dim == 1:
        dx = xvec[1] - xvec[0]  # Spatial step size

        for i in range(1, len(tvec)):
            if not finaltimeonly: 
                u_num = u[:, i]
            else:
                u_num = u
            # Compute exact solution at time tvec[i]
            u_ex_mat = exact_solution(xvec, tvec[i])
            # Compute the error
            err = u_ex_mat - u_num

            if np.isinf(q):
                # Compute max norm (L-infinity norm)
                err_q_x[i - 1] = np.max(np.abs(err)) / np.max(np.abs(u_num))
            else:
                # Compute Lq norm
                err_q_x[i - 1] = (dx * np.sum(np.abs(err)**q))**(1/q) / \
                    ((dx * np.sum(np.abs(u_ex_mat)**q))**(1/q))
    else:
        # 2D or 3D domains
        raise NotImplementedError

    return err_q_x