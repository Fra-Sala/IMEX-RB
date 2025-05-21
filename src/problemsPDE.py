import os
import numpy as np
import scipy.sparse as sp
from functools import cached_property
from abc import ABC

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PDEBase(ABC):
    def __init__(self, shape, lengths, sdim, bc_funcs=None, forcing=None, is_linear=False):
        """Initializer of the PDE base class."""

        self.shape = tuple(shape)    # natural order (Nx, Ny, Nz)
        self.lengths = tuple(lengths)
        self.ndim = len(self.shape)
        self.soldim = sdim
        self.Nh = int(np.prod(self.shape)) * sdim

        self.P = np.empty(0)

        # grid spacing
        self.dx = [self.lengths[i] / (self.shape[i] - 1)
                   for i in range(self.ndim)]

        # build axes in x, y, z, ... order
        axes = [np.linspace(0, self.lengths[i], self.shape[i])
                for i in range(self.ndim)]
        # meshgrid with default 'xy' ordering: coords[d].shape = (Ny, Nx, Nz)
        self.coords = np.meshgrid(*axes, indexing='xy')
        # Store (Ny, Nx, Nz)
        self.pyshape = self.coords[0].shape
        self.idxs = np.indices(self.pyshape)

        if bc_funcs is None:
            bc_funcs = [0.0] * (2 * self.ndim)
        assert len(bc_funcs) == 2 * self.ndim, \
            f"Invalid number of boundary conditions. Expected {2*self.ndim}, got {len(bc_funcs)}"

        # We now create a list of the boundary Dirichlet data in the form
        # bc_funcs(t) = [bc_xmin(t), bc_xmax(t), bc_ymin(t), etc]
        # where each entry of the list already returns a vector of the
        # evaluations over that boundary (a few nodes of the grid)
        self.bc_funcs = []
        for user_bc, (face_coords, face_id) in \
                zip(bc_funcs, self._boundary_info()):
            self.bc_funcs.append(self.wrap_bc(user_bc, face_coords))

        # Forcing term
        self.f = forcing

        # Assemble the stencils for the given problem
        self.assemble_stencil()

        self.is_linear = is_linear

        return

    def _boundary_info(self):
        """
        Generator function that yields boundary information.

        For each dimension of the grid (x,y,z), this function iterates over
        the two boundary sides (xmin, xmax, ymin, ymax, etc) and extracts
        the coordinates of the boundary faces.

        Yields:
            tuple: A tuple containing:
                - face_coords (list): A list of flattened arrays representing
                  the coordinates of the boundary face for each dimension.
                - boundary_info (tuple): A tuple (d, side) where:
                    - d (int): The dimension index of the boundary.
                    - side (int): The side of the boundary (0 for the start,
                      -1 for the end).
        """

        for d in range(self.ndim):
            a = 1 if (d == 0 and self.ndim > 1) else 0 if d == 1 else d
            for side in (0, -1):
                slicer = [slice(None)] * self.ndim
                slicer[a] = side
                sl = tuple(slicer)
                face_coords = [C[sl].flatten() for C in self.coords]
                yield face_coords, (d, side)

        return

    def wrap_bc(self, user_bc, coords):
        """
        Turn a user_bc for one face into bc(t) -> flat array of length
        soldim * N_face.  coords is a list of nd arrays of length N_face.
        Fallback: use the implemented exact_solution to infer the Dirichlet
        boundary conditions.
        """

        N = coords[0].size
        s = self.soldim

        if callable(user_bc):
            # user_bc(t, *coords) -> array shape (sdim, N)
            return lambda t: np.concatenate(user_bc(t, *coords))

        if user_bc is not None:
            vals = np.atleast_1d(user_bc)
            if vals.size == 1:
                vals = np.full(s, vals.item())
            if vals.size != s:
                raise ValueError(f"Expected {s} constants, got {vals.size}")
            return lambda t: np.hstack([np.full(N, vals[i]) for i in range(s)])

        # fallback: sample exact_solution on this face
        return lambda t: self.exact_solution(t, *coords)

    @cached_property
    def dirichlet_idx(self):
        """
        Compute the indices corresponding to the Dirichlet BCs.

        Returns:
            numpy.ndarray: A 1D array containing the global indices of the
            Dirichlet boundary points for all solution dimensions. The indices
            are computed for a flattened representation of the grid.
        """

        mask = np.zeros(self.pyshape, bool)
        for ax, sz in enumerate(self.pyshape):
            idx = self.idxs[ax]
            mask |= (idx == 0) | (idx == sz - 1)
        base = np.nonzero(mask.flatten())[0]
        Nf = int(np.prod(self.pyshape))

        return np.concatenate([base + k * Nf for k in range(self.soldim)])

    @cached_property
    def free_idx(self):
        """
        Compute the indices of non-Dirichlet boundary nodes.

        Returns:
            numpy.ndarray: An array of indices corresponding to non-Dirichlet
            boundary nodes.
        """
        return np.setdiff1d(np.arange(self.Nh),
                            self.dirichlet_idx,
                            assume_unique=True)

    def lift_vals(self, t):
        """
        Constructs a full solution vector of length `Nh` that incorporates
        Dirichlet boundary conditions (BCs).

        This method handles boundary conditions for 1D, 2D, and 3D problems,
        reshaping and assigning the boundary values.

        Args:
            t (float): The current time.

        Returns:
            np.ndarray: A 1D array of length `Nh`, with 0s at non-Dirichlet
            entries.
        """
        # component‐first grid array
        comps = np.zeros((self.soldim,) + self.pyshape)
        if self.ndim == 1:
            dim_order = range(self.ndim)

        elif self.ndim == 2:
            dim_order = (1, 0)

        elif self.ndim == 3:
            dim_order = (1, 0, 2)

        for ax_bc, ax in enumerate(dim_order):
            # Using ax_bc = 0, ax = 1, we start from the x limits
            idx = self.idxs[ax]
            low_mask = (idx == 0)
            high_mask = (idx == self.pyshape[ax] - 1)

            bc_low = self.bc_funcs[2*ax_bc](t)
            bc_high = self.bc_funcs[2*ax_bc+1](t)
            # reshape into (soldim, N_edge)
            N_edge = bc_low.size // self.soldim
            bc_low = bc_low.reshape((self.soldim, N_edge))
            bc_high = bc_high.reshape((self.soldim, N_edge))

            for comp in range(self.soldim):
                comps[comp][low_mask] = bc_low[comp]
                comps[comp][high_mask] = bc_high[comp]

        # flatten component‐first into 1D [u; v; …]
        return comps.reshape(-1)

    def assemble_stencil(self):
        """
        Placeholder for assembling finite difference stencil.
        NOTE: This method is not implemented and should be overridden in a subclass.
        """
        pass

    @staticmethod
    def laplacian(dim):
        """
        Returns tridiagonal approximation of 1D Laplacian.
        """
        e = np.ones(dim)
        return sp.diags([e, -2*e, e], offsets=[-1, 0, 1],
                        shape=(dim, dim), format='csr')

    @staticmethod
    def advection_centered(dim):
        """
        Return tridiagonal matrix for approximation of 
        first derivative in space (advection) using a centered
        scheme.
        """
        e = np.ones(dim)
        return sp.diags([-e, np.zeros(dim), e], offsets=[-1, 0, 1],
                        shape=(dim, dim), format='csr')
    
    @staticmethod
    def advection_upwind(dim):
        """
        Return a matrix for approximating the first derivative in space
        (advection)
        using a first-order upwind (backward difference) scheme. This
        implementation assumes a positive flow velocity.
        """
        # Positive velocities
        e = np.ones(dim)
        return sp.diags([-e, e], offsets=[-1, 0],
                        shape=(dim, dim), format='csr')

    def initial_condition(self, t0=0.0):
        """
        Computes the initial condition of the PDE at the given initial time.
        Can be overridden if needed.
        """
        # For multi-component solutions, create in component-first ordering
        # but flatten in domain-first ordering
        if hasattr(self, 'exact_solution'):
            # Flatten coordinate arrays - these already have correct ordering
            coords_flat = [c.flatten() for c in self.coords]
            # Get solution values at all points
            u0 = self.exact_solution(t0, *coords_flat)
            # Ensure we're returning in correct flattened order
            return u0
        else:
            return np.zeros(self.Nh)

    def rhs(self, *args):
        """
        Compute the right-hand side (RHS) of the PDE system.
        NOTE: This method is not implemented and should be overridden in a subclass.
        """
        pass

    def jacobian(self, *args):
        """
        Compute the Jacobian of the RHS of the PDE system.
        NOTE: This method is not implemented and should be overridden in a subclass.
        """
        pass

    def rhs_free(self, t, u0):
        """
        Evaluate rhs for the *free* components only
        """
        uL = self.lift_vals(t)
        full = uL.copy()
        full[self.free_idx] = u0

        rhs_full = self.rhs(t, full)

        return rhs_full[self.free_idx]

    def jacobian_free(self, t, u):
        """
        Build the Jacobian d(rhs_free)/d(u0),
        i.e. get rid of the Dirichlet rows and columns
        of the Jacobian after its evaluation.
        """
        J_full = self.jacobian(t, u)
        Jfree = J_full[self.free_idx, :][:, self.free_idx]
        return Jfree

    # TO DO: source term NEVER tested
    # TO DO: avoid recomputing the source term in IMEX-RB?
    # @cached_property
    def source_term(self, t):
        """
        Evaluate the forcing term self._f over the interior of the domain,
        zero on the Dirichlet boundary, for any ndim and soldim.
        Returns a flat array of length Nh with component‐first ordering.
        """

        if self.f is None:
            return np.zeros(self.Nh)

        F = np.zeros((self.soldim,) + self.pyshape)
        interior = tuple(slice(1, -1) for _ in range(self.ndim))

        # Retrieve 'interior' grids for all meshgrids
        meshes_int = [C[interior] for C in self.coords]
        vals = np.asarray(self.f(t, *meshes_int))

        for comp in range(self.soldim):
            F[(comp,) + interior] = vals[comp]

        return F.reshape(-1)

    def exact_solution(self, *args):
        """
        Return, if available, the exact solution to the problem.
        NOTE: This method is not implemented and should be overridden in a subclass.
        """
        pass

    def preconditioner(self, matrix):
        """
        Constructs and returns a preconditioner matrix for use in iterative
        solvers.

        Parameters:
        -----------
        matrix : scipy.sparse.spmatrix
            The input sparse matrix for which the preconditioner is to be
            constructed.

        Returns:
        --------
        scipy.sparse.linalg.LinearOperator
            A linear operator representing the inverse of the preconditioner,
            which can be used by GMRES.
        """
        # For instance: extract the three diagonals
        d0 = matrix.diagonal(0)
        d1 = matrix.diagonal(1)
        d_1 = matrix.diagonal(-1)
        self.P = sp.diags([d_1, d0, d1], offsets=[-1, 0, 1], format='csr')

        # Define inverse of preconditioner self.P
        M_x = (lambda x: sp.linalg.spsolve(self.P, x))

        # Define operator object for scipy GMRES
        return sp.linalg.LinearOperator(matrix.shape, M_x)


class Burgers2D(PDEBase):
    """
    Class to solve the 2D vectorial viscous Burgers problem.
    It should implement a rhs method returning the evaluation
    of \\mu A x - C(x)x for a given x.
    """
    def __init__(self, Nx, Ny, Lx, Ly, mu=0.1):
        # 2D grid (shape: [Nx, Ny])
        shape = (Nx, Ny)
        lengths = (Lx, Ly)
        sdim = 2

        # Pay attention to the order of BC list: first the ones defined by
        # xlim (left, right),
        # then the ones defined by ylim (bottom, top), etc
        # [bc_left, bc_right, bc_bottom, bc_top]
        bc_list = [None] * 4

        self.mu = mu

        self.A_diff = np.empty(0)
        self.Adv_x = np.empty(0)
        self.Adv_y = np.empty(0)

        # By setting them to None, we are taking them from the exact sol
        super().__init__(shape, lengths, sdim, bc_funcs=bc_list)

        return

    def assemble_stencil(self):
        """
        Override parent class. Stencil for 2D Burgers.
        """
        Nx, Ny = self.shape
        dx, dy = self.dx

        # Diffusion operators (Laplacian) via central differences
        D2x = self.laplacian(Nx)
        D2y = self.laplacian(Ny)
        Ix = sp.eye(Nx, format='csr')
        Iy = sp.eye(Ny, format='csr')
        Lx = (self.mu / dx**2) * D2x
        Ly = (self.mu / dy**2) * D2y

        # Assemble diffusion matrix for a single vel component
        A_diff_scalar = sp.kron(Iy, Lx, format='csr') + \
            sp.kron(Ly, Ix, format='csr')

        # Block structure of A_diff for vector problem: [u; v]
        A_diff = sp.block_diag([A_diff_scalar, A_diff_scalar], format='csr')

        # Advection operators via central differences
        Cx = self.advection_centered(Nx) / (2 * dx)
        Cy = self.advection_centered(Ny) / (2 * dy)
        Adv_x = sp.kron(Iy, Cx, format='csr')
        Adv_y = sp.kron(Cy, Ix, format='csr')

        # Store operators for use in rhs and C(x)
        self.A_diff = A_diff
        self.Adv_x = Adv_x
        self.Adv_y = Adv_y

        return

    def rhs(self, t, x):
        """
        Return the evaluation of RHS of the Cauchy problem.
        """
        return self.A_diff @ x - self._C_adv(x) @ x + self.source_term(t)

    def _C_adv(self, x):
        """
        Build the nonlinear convection matrix C(x) = [u; v] · ∇.
        """
        n = self.A_diff.shape[0] // 2
        # split velocity vector
        u = x[:n]
        v = x[n:]
        # elementwise scale advective stencils
        Ux = sp.diags(u) @ self.Adv_x
        Vy = sp.diags(v) @ self.Adv_y
        conv = Ux + Vy
        # expand to block for [u; v]
        C = sp.block_diag([conv, conv], format='csr')

        return C

    def jacobian(self, t, x):
        """
        Assemble the Jacobian J = d/dx [A_diff*x - C(x)x] = -C(x) - dC(x)x
                                + A_diff.
        """
        # split components
        n = self.A_diff.shape[0]//2
        u = x[:n]
        v = x[n:]

        # precompute advective products
        Adv_x_u = self.Adv_x @ u
        Adv_y_u = self.Adv_y @ u
        Adv_x_v = self.Adv_x @ v
        Adv_y_v = self.Adv_y @ v

        diag_xu = sp.diags(Adv_x_u, format='csr')
        diag_yu = sp.diags(Adv_y_u, format='csr')
        diag_xv = sp.diags(Adv_x_v, format='csr')
        diag_yv = sp.diags(Adv_y_v, format='csr')

        # convection Jacobian block
        top = diag_xu + sp.diags(u) @ self.Adv_x + sp.diags(v) @ self.Adv_y
        bottom = diag_yv + sp.diags(u) @ self.Adv_x + sp.diags(v) @ self.Adv_y
        J11 = top
        J12 = diag_yu
        J21 = diag_xv
        J22 = bottom
        J_conv = sp.block_array([[J11, J12], [J21, J22]], format='csr')

        # full Jacobian
        J = -J_conv + self.A_diff

        return J

    def exact_solution(self, t, x, y):
        """
        2D Burgers exact solution:

            u(x, y, t) = 3/4 - 1/4 * [1 + exp((-4*x + 4*y - t)*Re/32)]^{-1}
            v(x, y, t) = 3/4 + 1/4 * [1 + exp((-4*x + 4*y - t)*Re/32)]^{-1}

        where mu = 1/Re.
        Returns a concatenated vector of u followed by v (column-wise).
        """

        Re = 1.0 / self.mu
        arg = (-4.0 * x + 4.0 * y - t) * Re / 32.0
        exp_arg = np.exp(arg)
        inv = 1.0 / (1.0 + exp_arg)

        u_ex = 0.75 - 0.25 * inv
        v_ex = 0.75 + 0.25 * inv

        return np.concatenate([u_ex.flatten(), v_ex.flatten()])


class Heat1D(PDEBase):
    def __init__(self, Nx, Lx, mu=0.1):
        """Initializer of the Heat equation 1D class"""

        shape = (Nx,)
        lengths = (Lx,)
        bc_list = [None] * 2
        sdim = 1  # scalar problem

        self.mu = mu

        self.A = np.empty(0)

        super().__init__(shape, lengths, sdim, bc_funcs=bc_list, is_linear=True)

        self.domain = self.coords[0]

        return

    def assemble_stencil(self):
        dx = self.dx[0]
        self.A = self.mu * self.laplacian(self.Nh) / (dx ** 2)
        return

    def rhs(self, t, u):
        return self.A * u

    def jacobian(self, t, u):
        return self.A

    def exact_solution(self, t, x, lam=np.pi):
        return np.sin(lam * x) * np.exp(-self.mu * lam**2 * t)


class Heat2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, mu=1, sigma=1, center=None):
        """Initializer of the Heat equation 2D class"""

        shape = (Nx, Ny) 
        lengths = (Lx, Ly)
        bc_list = [None] * 4
        sdim = 1  # scalar problem

        self.mu = mu
        self.sigma = sigma
        self.center = np.array([Lx//2, Ly//2]) if center is None else np.array(center)

        self.A = np.empty(0)

        forcing = None  # TODO: update according to latest approach

        super().__init__(shape, lengths, sdim, bc_funcs=bc_list, forcing=forcing, is_linear=False)

        return

    def assemble_stencil(self):
        """
        Build the 2D diffusion operator using Kronecker products:
        A = κ*(I_y ⊗ D₂ₓ + D₂ᵧ ⊗ I_x)
        where D₂ₓ, D₂ᵧ are 1D second‐derivative matrices.
        """

        Nx, Ny = self.shape
        dx, dy = self.dx

        # Diffusion operators (Laplacian) via central differences
        D2x = self.laplacian(Nx)
        D2y = self.laplacian(Ny)
        Ix = sp.eye(Nx, format='csr')
        Iy = sp.eye(Ny, format='csr')
        Lx = (self.mu / dx**2) * D2x
        Ly = (self.mu / dy**2) * D2y

        # Assemble diffusion matrix for a single vel component
        self.A = sp.kron(Iy, Lx, format='csr') + sp.kron(Ly, Ix, format='csr')

        return

    def rhs(self, t, u):
        return self.A * u + self.source_term(t)

    def jacobian(self, t, u):
        return self.A

    def exact_solution(self, t, x, y):
        """
        Exact solution to the problem
        """

        # TODO: update according to latest approach

        factor = 1 / (2 * np.pi * (self.sigma**2 + 2 * self.mu * t))
        exponent = -((x - self.center[0])**2 + (y - self.center[1])**2) / (2 * (self.sigma**2 + 2 * self.mu * t))
        sol = factor * np.exp(exponent)

        return sol


class AdvDiff2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, mu=1, sigma=1, vx=1, vy=1, center=None):
        """Initializer of the Advection-diffusion equation 2D class"""

        shape = (Nx, Ny)
        lengths = (Lx, Ly)
        bc_list = [None, None, None, None]
        sdim = 1  # scalar problem

        self.mu = mu
        self.sigma = sigma
        self.center = np.array([Lx // 2, Ly // 2]) if center is None else np.array(center)
        self.vx = vx
        self.vy = vy

        self.A = np.empty(0)

        super().__init__(shape, lengths, sdim, bc_funcs=bc_list, is_linear=True)

        return

    def assemble_stencil(self):
        Nx, Ny = self.shape
        dx, dy = self.dx

        # Diffusion operators (Laplacian) via central differences
        D2x = self.laplacian(Nx)
        D2y = self.laplacian(Ny)
        Ix = sp.eye(Nx, format='csr')
        Iy = sp.eye(Ny, format='csr')
        Lx = (self.mu / dx**2) * D2x
        Ly = (self.mu / dy**2) * D2y
        # Assemble diffusion matrix
        A_diff = sp.kron(Iy, Lx, format='csr') + sp.kron(Ly, Ix, format='csr')

        # Advection operators via upwind
        # Cx = self.advection_upwind(Nx) / (dx)
        # Cy = self.advection_upwind(Ny) / (dy)
        # Advection operators via centered scheme
        Cx = self.advection_centered(Nx) / (2 * dx)
        Cy = self.advection_centered(Ny) / (2 * dy)
        Adv_x = sp.kron(Iy, Cx, format='csr')
        Adv_y = sp.kron(Cy, Ix, format='csr')

        # Store operators for use in rhs and C(x)
        A_x = self.vx * Adv_x
        A_y = self.vy * Adv_y

        # total operator A: diffusion - (vx ∂/∂x + vy ∂/∂y)
        self.A = A_diff - (A_x + A_y)

        return

    def rhs(self, t, u):
        return self.A * u

    def jacobian(self, t, u):
        return self.A

    def exact_solution(self, t, x, y):
        """
        Exact solution to the problem
        """

        # TODO: update according to latest approach

        factor = 1 / (2 * np.pi * (self.sigma**2 + 2*self.mu*t))
        exponent = -((x - self.center[0] - self.vx*t)**2 + (y - self.center[1] - self.vy*t)**2) / \
                   (2 * (self.sigma**2 + 2*self.mu*t))
        sol = factor * np.exp(exponent)

        return sol
