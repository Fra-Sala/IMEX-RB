import numpy as np


def compute_errors(u, tvec, coords, exact_solution, dim=1, q=2,
                   finaltimeonly=False):
    """
    Compute FD errors over a dim-dimensional domain.
    coords: list of length dim; for dim=1, coords=[x];
            for dim=2, coords=[X, Y] mesh‐grids as from PDE.coords
    """
    if finaltimeonly:
        tvec = tvec[-2:]
        u = u[..., -1]

    nsteps = len(tvec) - 1
    err_q = np.zeros(nsteps)

    for i in range(1, len(tvec)):
        t = tvec[i]
        u_num = u[..., i] if not finaltimeonly else u

        if dim == 1:
            x = coords[0]
            dx = x[1] - x[0]
            u_ex = exact_solution(x, t)
            err = u_ex - u_num

            if np.isinf(q):
                err_q[i-1] = np.max(np.abs(err)) / np.max(np.abs(u_num))
            else:
                norm_err = (dx * np.sum(np.abs(err)**q))**(1/q)
                norm_ex = (dx * np.sum(np.abs(u_ex)**q))**(1/q)
                err_q[i-1] = norm_err / norm_ex

        elif dim == 2:
            X, Y = coords
            # assume uniform spacing
            dx = X[1, 0] - X[0, 0]
            dy = Y[0, 1] - Y[0, 0]
            u_ex = exact_solution(X, Y, t).flatten()
            err = u_ex - u_num

            if np.isinf(q):
                err_q[i-1] = np.max(np.abs(err)) / np.max(np.abs(u_num))
            else:
                area = dx * dy
                norm_err = (area * np.sum(np.abs(err)**q))**(1/q)
                norm_ex = (area * np.sum(np.abs(u_ex)**q))**(1/q)
                err_q[i-1] = norm_err / norm_ex

        else:
            raise NotImplementedError(f"dim={dim} not supported")

    return err_q
