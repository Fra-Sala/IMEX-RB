import numpy as np
import scipy.sparse as sp
from functools import cached_property


class PDEBase:
    """
    Abstract base for PDEs in any dimension with Dirichlet BCs.
    Subclasses must implement:
      - assemble_stencil()
      - source_term(t)
      - exact_solution(..., t)
    """
    def __init__(self, shape, lengths, kappa, bc_funcs):
        self.shape = tuple(shape)
        self.lengths = tuple(lengths)
        self.kappa = kappa
        self.ndim = len(self.shape)
        assert len(bc_funcs) == 2 * self.ndim
        # Coercion: Ensure bc_funcs are callables of t
        self.bc_funcs = [f if callable(f) else (lambda t, val=f: val)
                         for f in bc_funcs]
        self.Nh = int(np.prod(self.shape))
        # Grid spacing and index/coordinate arrays
        self.dx = [self.lengths[i] / (self.shape[i] - 1)
                   for i in range(self.ndim)]
        # A list of indices with shape (ndims, Nx, Ny, Nz)
        # to be used to create self.coords (similar to meshgrid)
        self.idxs = np.indices(self.shape)
        # The following is a list of the result of "meshgrid"
        self.coords = [self.idxs[i] * self.dx[i] for i in range(self.ndim)]
        self.A = None
        self.assemble_stencil()

    @cached_property
    def dirichlet_indices(self):
        """
        Return True for all indices corresponding to Dirichlet BCs.
        This is a cached property: it is computed once when assembling
        the object, and then retrieved from memory.
        """
        mask = np.zeros(self.shape, dtype=bool)
        for i, size in enumerate(self.shape):
            idx = self.idxs[i]
            mask |= (idx == 0) | (idx == size - 1)
        return mask.flatten()  # np.nonzero(mask.flatten())[0]

    def enforce_bcs(self, v, t):
        """
        Apply lower/upper boundary conditions
        across the d dimensions (1,2,3). This because
        self.bc_funcs is supposed to contain bcx_min, bcx_max,
        bcy_min, bcy_max, bcz_min, bcz_max.
        """
        # Take a flatten array, and go back to
        # a shape compatible with the grid
        v = v.reshape(self.shape)
        for i in range(self.ndim):
            # Retrieve idx with shape (Nx, Ny, Nz)
            idx = self.idxs[i]
            low = (idx == 0)
            high = (idx == self.shape[i] - 1)
            v[low] = self.bc_funcs[2*i](t)
            v[high] = self.bc_funcs[2*i+1](t)
        return v.flatten()

    def apply_bc(self, M, b, t):
        """
        We enforce the Dirichlet bcs for an implicit
        scheme in the form:
                        M*x = b
        by setting the Dirichlet rows of M to identity
        and the same rows of b to the Dirichlet value.
        """
        uD = np.zeros(self.Nh)
        uD = self.enforce_bcs(uD, t)
        b_lifted = b  # - M @ uD
        M_bc = M.tolil()
        rows = self.dirichlet_indices
        M_bc[rows, :] = 0
        M_bc[rows, rows] = 1.0
        b_lifted[rows] = uD[rows]
        return M_bc.tocsr(), b_lifted

    def initial_condition(self, t0=0.0):
        """
        Computes the initial condition of the PDE at the given initial time.

        Parameters:
            t0 (float, optional): The initial time at which the solution is
                evaluated.
                Defaults to 0.0.

        Returns:
            numpy.ndarray: A flattened array representing the initial condition
            of the PDE evaluated at the spatial grid.
        """
        args = [c.flatten() for c in self.coords]
        u0 = self.exact_solution(t0, *args)
        return u0.flatten()


class Heat1D(PDEBase):
    def __init__(self, N, L, kappa, bc_left=None, bc_right=None, f=None):
        shape = (N,)
        lengths = (L,)
        super().__init__(shape, lengths, kappa, [None, None])
        self.domain = self.coords[0]
        self._f = f if f is not None else (lambda t, x: 0.0)
        # If no BC provided, use exact_solution at boundaries
        if bc_left is None:
            left = (lambda t: self.exact_solution(t, self.coords[0][0]))
        else:
            left = bc_left if callable(bc_left) else (lambda t: bc_left)
        if bc_right is None:
            right = (lambda t: self.exact_solution(t, self.coords[0][-1]))
        else:
            right = bc_right if callable(bc_right) else (lambda t: bc_right)
        self.bc_funcs = [left, right]

    def assemble_stencil(self):
        coef = self.kappa / (self.dx[0] ** 2)
        diags = [coef * np.ones(self.Nh - 1),
                 -2 * coef * np.ones(self.Nh),
                 coef * np.ones(self.Nh - 1)]
        self.A = sp.diags(diags, (-1, 0, 1), shape=(self.Nh, self.Nh),
                          format='csr')

    def source_term(self, t):
        f_vec = np.zeros(self.shape)
        f_vec[1:-1] = self._f(t, self.coords[0][1:-1])
        return f_vec.flatten()

    def exact_solution(self, t, x, lam=np.pi):
        return np.sin(lam * x) * np.exp(-self.kappa * lam**2 * t)


class Heat2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, kappa,
                 bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=0.0, f=None):
        # 2D grid (shape: [Ny, Nx])
        shape = (Ny, Nx)
        lengths = (Ly, Lx)
        # bc_funcs order: [y_low, y_high, x_low, x_high]
        bottom = bc_bottom if callable(bc_bottom) else (lambda t: bc_bottom)
        top = bc_top if callable(bc_top) else (lambda t: bc_top)
        left = bc_left if callable(bc_left) else (lambda t: bc_left)
        right = bc_right if callable(bc_right) else (lambda t: bc_right)
        super().__init__(shape, lengths, kappa, [left, right, bottom, top])
        self._f = f if f is not None else (lambda t, x, y: 0.0)

    def assemble_stencil(self):
        """
        Build the 2D diffusion operator using Kronecker products:
        A = κ*(I_y ⊗ D₂ₓ + D₂ᵧ ⊗ I_x)
        where D₂ₓ, D₂ᵧ are 1D second‐derivative matrices.
        """
        Ny, Nx = self.shape
        dx, dy = self.dx
        # 1D second‐derivative stencils
        ex = np.ones(Nx)
        ey = np.ones(Ny)
        D2x = sp.diags([ex, -2*ex, ex], offsets=[-1, 0, 1], shape=(Nx, Nx))
        D2y = sp.diags([ey, -2*ey, ey], offsets=[-1, 0, 1], shape=(Ny, Ny))
        Ix = sp.eye(Nx)
        Iy = sp.eye(Ny)
        Lx = (self.kappa / dx**2) * D2x
        Ly = (self.kappa / dy**2) * D2y
        # 2D Laplacian via kron
        A = sp.kron(Iy, Lx, format='csr') + sp.kron(Ly, Ix, format='csr')
        self.A = A

    def source_term(self, t):
        F = np.zeros(self.shape)
        interior = (slice(1, -1), slice(1, -1))
        F[interior] = self._f(t, *[mesh[interior] for mesh in self.coords])
        return F.flatten()

    def exact_solution(self, t, x, y, lamx=np.pi, lamy=np.pi):
        """
        2D separable exact solution:
        u(x,y,t) = sin(lamx * x) * sin(lamy * y) *
                   exp(-kappa*(lamx^2 + lamy^2)*t)
        """
        return (np.sin(lamx * x) *
                np.sin(lamy * y) *
                np.exp(-self.kappa * (lamx**2 + lamy**2) * t))
    

class AdvDiff2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, kappa, vx, vy,
                 bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=0.0, f=None):
        # 2D grid (shape: [Ny, Nx])
        shape = (Ny, Nx)
        lengths = (Ly, Lx)
        xs = np.linspace(0, Lx, Nx)
        ys = np.linspace(0, Ly, Ny)

        # Define evaluations of BCs
        def wrap_bc(user_bc, s_vals):
            if callable(user_bc):
                return lambda t: user_bc(t, s_vals)
            elif user_bc is not None:
                return lambda t: user_bc
            else:
                return lambda t: self.exact_solution(t, *s_vals)

        # bottom (y=0)
        bottom_x = xs
        bottom_y = np.zeros_like(xs)
        bottom = wrap_bc(bc_bottom, (bottom_x, bottom_y))

        # top (y=Ly)
        top_x = xs
        top_y = Ly * np.ones_like(xs)
        top = wrap_bc(bc_top, (top_x, top_y))

        # left (x=0)
        left_y = ys
        left_x = np.zeros_like(ys)
        left = wrap_bc(bc_left, (left_x, left_y))

        # right (x=Lx)
        right_y = ys
        right_x = Lx * np.ones_like(ys)
        right = wrap_bc(bc_right, (right_x, right_y))

        # velocity field (scalars)
        self.vx = vx
        self.vy = vy
        # Pay attention to the order of BC list: first the ones defined by
        # xlim (left, right),
        # then the ones defined by ylim (bottom, top)
        super().__init__(shape, lengths, kappa, [left, right, bottom, top])

        self._f = f if f is not None else (lambda t, x, y: 0.0)

    def assemble_stencil(self):
        Ny, Nx = self.shape
        dx, dy = self.dx

        # diffusion part via Kron
        ex = np.ones(Nx)
        ey = np.ones(Ny)
        D2x = sp.diags([ex, -2*ex, ex], offsets=[-1, 0, 1], shape=(Nx, Nx))
        D2y = sp.diags([ey, -2*ey, ey], offsets=[-1, 0, 1], shape=(Ny, Ny))
        Ix = sp.eye(Nx)
        Iy = sp.eye(Ny)
        Lx = (self.kappa/dx**2) * D2x
        Ly = (self.kappa/dy**2) * D2y
        A_diff = sp.kron(Iy, Lx, format='csr') + sp.kron(Ly, Ix, format='csr')

        # advection part: upwind for positive velocities
        # 1D first‐derivative stencils
        D1x = sp.diags([-ex, ex], offsets=[-1, 0], shape=(Nx, Nx)) / (dx)
        D1y = sp.diags([-ey, ey], offsets=[-1, 0], shape=(Ny, Ny)) / (dy)
        A_adv_x = sp.kron(Iy, D1x, format='csr')
        A_adv_y = sp.kron(D1y, Ix, format='csr')
        
        # Multiply each 1st order spatial derivative by velocity (constant)
        A_x = self.vx * A_adv_x
        A_y = self.vy * A_adv_y

        # total operator A: diffusion - (vx ∂/∂x + vy ∂/∂y)
        self.A = A_diff - (A_x + A_y)

    def source_term(self, t):
        F = np.zeros(self.shape)
        interior = (slice(1, -1), slice(1, -1))
        F[interior] = self._f(t, *[mesh[interior] for mesh in self.coords])
        return F.flatten()

    def exact_solution(self, t, x, y):
        """
        2D advective-diffusive exact solution:

            u_ex(x, y, t) = 1/(4t + 1) * exp(-((x - c_x*t - 0.5)^2 +
                             (y - c_y*t - 0.5)^2) / (mu*(4t + 1)))

        where c_x and c_y are obtained from the velocity field components.
        mu is the diffusivity parameter.
        """
        c_x = self.vx
        c_y = self.vy
        factor = 1.0 / (4 * t + 1)
        exponent = -(((x - c_x * t - 0.5) ** 2) + ((y - c_y * t - 0.5) ** 2)) \
            / (self.kappa * (4 * t + 1))
        return factor * np.exp(exponent)

