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
        self.idxs = np.indices(self.shape)
        self.coords = [self.idxs[i] * self.dx[i] for i in range(self.ndim)]
        self.A = None
        self.assemble_stencil()

    @cached_property
    def dirichlet_indices(self):
        """
        Return True for all indices corresponding to Dirichlet BCs.
        """
        mask = np.zeros(self.shape, dtype=bool)
        for i, size in enumerate(self.shape):
            idx = self.idxs[i]
            mask |= (idx == 0) | (idx == size - 1)
        return np.nonzero(mask.flatten())[0]

    def enforce_bcs(self, v, t):
        # Take a flatten array, and go back to
        # a shape compatible with the grid
        v = v.reshape(self.shape)
        for i in range(self.ndim):
            idx = self.idxs[i]
            low = (idx == 0)
            high = (idx == self.shape[i] - 1)
            v[low] = self.bc_funcs[2*i](t)
            v[high] = self.bc_funcs[2*i+1](t)
        return v.flatten()

    def apply_bc(self, M, b, t):
        uD = np.zeros(self.Nh)
        uD = self.enforce_bcs(uD, t)
        b_lifted = b - M @ uD
        M_bc = M.tolil()
        rows = self.dirichlet_indices
        M_bc[rows, :] = 0
        M_bc[rows, rows] = 1.0
        b_lifted[rows] = uD[rows]
        return M_bc.tocsr(), b_lifted

    def initial_condition(self, t0=0.0):
        args = [c.flatten() for c in self.coords]
        u0 = self.exact_solution(*args, t0)
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
            left = (lambda t: self.exact_solution(self.coords[0][0], t))
        else:
            left = bc_left if callable(bc_left) else (lambda t: bc_left)
        if bc_right is None:
            right = (lambda t: self.exact_solution(self.coords[0][-1], t))
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

    def exact_solution(self, x, t, lam=np.pi):
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
        super().__init__(shape, lengths, kappa, [bottom, top, left, right])
        self.X, self.Y = np.meshgrid(
            np.linspace(0, Lx, Nx),
            np.linspace(0, Ly, Ny),
            indexing='xy'
        )
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
        f_mat = np.zeros(self.shape)
        interior = (slice(1, -1), slice(1, -1))
        f_mat[interior] = self._f(
            t,
            self.X[interior],
            self.Y[interior]
        )
        return f_mat.flatten()

    def exact_solution(self, x, y, t, lamx=np.pi, lamy=np.pi):
        """
        2D separable exact solution:
        u(x,y,t) = sin(lamx * x) * sin(lamy * y) *
                   exp(-kappa*(lamx^2 + lamy^2)*t)
        """
        return (np.sin(lamx * x) *
                np.sin(lamy * y) *
                np.exp(-self.kappa * (lamx**2 + lamy**2) * t))
