import numpy as np
import scipy.sparse as sp
from functools import cached_property


class PDEBase:
    """
    Abstract base for PDEs in any dimension with Dirichlet BCs.
    Subclasses must implement:
      - assemble_stencil()
      - source_term(t)
      - exact_solution(t, coords)
    """
    def __init__(self, shape, lengths, sdim, mu, bc_funcs):
        self.shape = tuple(shape)
        self.lengths = tuple(lengths)
        self.mu = mu
        self.ndim = len(self.shape)
        self.soldim = sdim  # number of sol components e.g. (u,v) -> 2
        assert len(bc_funcs) == 2 * self.ndim
        # Total number of unknowns
        self.Nh = int(np.prod(self.shape)) * self.soldim
        # grid setup
        # self.dx = [dx, dy, dz]
        self.dx = [self.lengths[i] / (self.shape[i] - 1)
                   for i in range(self.ndim)]
        self.idxs = np.indices(self.shape)
        self.coords = [self.idxs[i] * self.dx[i]
                       for i in range(self.ndim)]

        # prepare bc_funcs: for each face, wrap into bc(t)->flat vector
        self.bc_funcs = []
        for func, (coords, face_id) in zip(bc_funcs, self._boundary_info()):
            self.bc_funcs.append(self.wrap_bc(func, coords, face_id))

        self.assemble_stencil()

    def _boundary_info(self):
        """
        Yield pairs (coords_on_face, face_id) in order corresponding to
        user-supplied bc_funcs.  face_id is a tuple (axis, side)
        where side is 0 (lower) or -1 (upper).
        Example: 2D domain (Ny, Nx) = (3,5), if x \\in [0,1], y in [0,1]
        this will yield:
        [0, 0, 0, 0, 0] and [0.  , 0.25, 0.5 , 0.75, 1.  ]
        (i.e. the coordinate pairs of the left bound)
        Then:
        [1, 1, 1, 1, 1] and [0.  , 0.25, 0.5 , 0.75, 1.  ]
        (i.e. the coordinate pairs of the right bound)

        and then same for the y-bounds, and z-bounds
        """
        for axis in range(self.ndim):
            for side in (0, -1):
                # build a tuple of coordinate arrays restricted to this face
                slicer = []
                for d in range(self.ndim):
                    if d == axis:
                        slicer.append(side if side == 0 else -1)
                    else:
                        slicer.append(slice(None))
                slicer = tuple(slicer)
                # each coord[d][slicer] is a 1D array of length N_face
                face_coords = [self.coords[d][slicer].flatten()
                               for d in range(self.ndim)]
                yield face_coords, (axis, side)

    def wrap_bc(self, user_bc, coords, face_id):
        """
        Turn a user_bc for one face into bc(t) -> flat array of length
        soldim * N_face.  coords is a list of nd arrays of length N_face.
        """
        N = coords[0].size
        sdim = self.soldim

        if callable(user_bc):
            def bc(t):
                # user_bc must return shape (sdim, N)
                uv = user_bc(t, *coords)
                return np.concatenate([uv[i] for i in range(sdim)])
            return bc

        if user_bc is not None:
            # constant tuple/list of length soldim
            if len(user_bc) != sdim:
                raise ValueError(f"Expected {sdim} constants, got {len(user_bc)}")
            
            def bc(t):
                return np.hstack([np.full(N, user_bc[i])
                                  for i in range(sdim)])
            return bc

        # fallback: sample exact_solution on this face
        def bc(t):
            # we simply return [u_face; v_face; etc]
            uv_face = self.exact_solution(t, *coords)
            return uv_face

        return bc

    @cached_property
    def dirichlet_idx(self):
        """
        Return a 1D array of integer DOF indices corresponding to
        Dirichlet BCs for all 'soldim' components stacked as
        [u; v; ...].
        """
        # Build the scalar mask on the grid
        mask = np.zeros(self.shape, dtype=bool)
        for axis, size in enumerate(self.shape):
            idx = self.idxs[axis]
            mask |= (idx == 0) | (idx == size - 1)

        # Flatten and get base indices on a single field
        N_tot = int(np.prod(self.shape))
        base = np.nonzero(mask.flatten())[0]

        # Offset for each component
        #    (component 0 uses base,
        #     component 1 uses base + N_tot, etc.)
        all_idx = np.concatenate([
            base + comp * N_tot
            for comp in range(self.soldim)
        ])

        return all_idx

    @cached_property
    def non_dirichlet_idx(self):
        """
        Return the sorted array of DOF indices *not* on Dirichlet.
        """
        N_tot = self.soldim * int(np.prod(self.shape))
        all_idx = np.arange(N_tot)
        return np.setdiff1d(all_idx, self.dirichlet_idx, assume_unique=True)

    def compute_bcs(self, t):
        """
        Return a solution vector
        with Dirichlet BCs.
        """
        # total grid points
        # N_tot = int(np.prod(self.shape))
        sdim = self.soldim

        # # split into components and reshape to match
        # # domain shape
        # comps = [v[i * N_tot: (i + 1) * N_tot].reshape(self.shape)
        #          for i in range(sdim)]
        comps = np.zeros((sdim,) + self.shape)

        # apply each face BC
        for axis in range(self.ndim):
            idx = self.idxs[axis]
            low_mask = (idx == 0)
            high_mask = (idx == self.shape[axis] - 1)

            # get bc for this face: flat length sdim * N_edge
            bc_flat_low = self.bc_funcs[2 * axis](t)
            bc_flat_high = self.bc_funcs[2 * axis + 1](t)

            # reshape to (sdim, N_edge)
            N_edge = bc_flat_low.size // sdim
            bc_low = bc_flat_low.reshape((sdim, N_edge))
            bc_high = bc_flat_high.reshape((sdim, N_edge))

            # assign for each component
            for comp in range(sdim):
                comps[comp][low_mask] = bc_low[comp]
                comps[comp][high_mask] = bc_high[comp]

        # re-concatenate and flatten
        return np.hstack([c.flatten() for c in comps])

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
    
    @cached_property
    def free_idx(self):
        """Indices of the unknowns *not* on Dirichlet faces."""
        return self.non_dirichlet_idx

    def lift(self, t):
        """
        Build the full-length vector u_L(t) that
        is zero in the interior and equals the BCs on the boundary.
        """
        return self.compute_bcs(t)

    def rhs_free(self, t, u0):
        """
        Evaluate rhs for the *free* components only:
        e.g. :
        du0/dt =   A_diff (u_L + u_0)
                 − C(u_L + u_0)(u_L + u_0)
                 + source
        then restrict to free_idx.
        """
        uL = self.lift(t)
        full = uL.copy()
        full[self.free_idx] += u0

        rhs_full = self.rhs(t, full)

        return rhs_full[self.free_idx]

    def jacobian_free(self, t, u):
        """
        Build the Jacobian d(rhs_free)/d(u0):
        it is the restriction of the full Jacobian
        at u = u_L + u_0 to the free rows/cols.
        """
        J_full = self.jacobian(t, u)
        Jfree = J_full[self.free_idx, :][:, self.free_idx]
        return Jfree

    # @cached_property
    def preconditioner(self, Mmod):
        """
        Returns a preconditioner matrix built from the tridiagonal of the
        system matrix (modified to deal with BCs).
        This takes the main diagonal and the first lower and upper diagonals.
        """
        # Extract the three diagonals
        d0 = Mmod.diagonal(0)
        d1 = Mmod.diagonal(1)
        d_1 = Mmod.diagonal(-1)
        self.P = sp.diags([d_1, d0, d1], offsets=[-1, 0, 1], format='csr')
        # Define inverse of preconditioner self.P
        M_x = (lambda x: sp.linalg.spsolve(self.P, x))
        # Define operator object for scipy GMRES
        return sp.linalg.LinearOperator(Mmod.shape, M_x)


class Heat1D(PDEBase):
    def __init__(self, N, L, mu, bc_left=None, bc_right=None, f=None):
        shape = (N,)
        lengths = (L,)
        super().__init__(shape, lengths, mu, [None, None])
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
        coef = self.mu / (self.dx[0] ** 2)
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
        return np.sin(lam * x) * np.exp(-self.mu * lam**2 * t)


class Heat2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, mu,
                 bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=0.0, f=None):
        # 2D grid (shape: [Ny, Nx])
        shape = (Ny, Nx)  # C-order for Python
        lengths = (Ly, Lx)
        # bc_funcs order: [y_low, y_high, x_low, x_high]
        bottom = bc_bottom if callable(bc_bottom) else (lambda t: bc_bottom)
        top = bc_top if callable(bc_top) else (lambda t: bc_top)
        left = bc_left if callable(bc_left) else (lambda t: bc_left)
        right = bc_right if callable(bc_right) else (lambda t: bc_right)
        super().__init__(shape, lengths, mu, [left, right, bottom, top])
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
        Lx = (self.mu / dx**2) * D2x
        Ly = (self.mu / dy**2) * D2y
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
                   exp(-mu*(lamx^2 + lamy^2)*t)
        """
        return (np.sin(lamx * x) *
                np.sin(lamy * y) *
                np.exp(-self.mu * (lamx**2 + lamy**2) * t))
    

class AdvDiff2D(PDEBase):
    def __init__(self, Nx, Ny, Lx, Ly, mu, vx, vy,
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
        super().__init__(shape, lengths, mu, [left, right, bottom, top])

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
        Lx = (self.mu/dx**2) * D2x
        Ly = (self.mu/dy**2) * D2y
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
            / (self.mu * (4 * t + 1))
        return factor * np.exp(exponent)


class Burgers2D(PDEBase):
    """
    Class to solve the 2D vectorial viscous Burgers problem.
    It should implement a rhs method returning the evaluation
    of \\mu A x - C(x)x for a given x.
    """
    def __init__(self, Nx, Ny, Lx, Ly, mu,
                 bc_left=None, bc_right=None,
                 bc_bottom=None, bc_top=None, f=None):
        # 2D grid (shape: [Ny, Nx])
        shape = (Ny, Nx)
        lengths = (Ly, Lx)
        sdim = 2
        # Pay attention to the order of BC list: first the ones defined by
        # xlim (left, right),
        # then the ones defined by ylim (bottom, top)
        super().__init__(shape, lengths, sdim, mu,
                         [bc_left, bc_right, bc_bottom, bc_top])

        self._f = f if f is not None else \
            (lambda t, x, y: np.zeros((self.soldim, *x.shape)))

    def assemble_stencil(self):
        Ny, Nx = self.shape      
        dx, dy = self.dx

        # Diffusion operators (Laplacian) via central differences
        ex = np.ones(Nx)
        ey = np.ones(Ny)
        D2x = sp.diags([ex, -2*ex, ex], offsets=[-1, 0, 1],
                       shape=(Nx, Nx), format='csr')
        D2y = sp.diags([ey, -2*ey, ey], offsets=[-1, 0, 1],
                       shape=(Ny, Ny), format='csr')
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
        Cx = sp.diags([-ex, np.zeros(Nx), ex], offsets=[-1, 0, 1],
                      shape=(Nx, Nx), format='csr') / (2*dx)
        Cy = sp.diags([-ey, np.zeros(Ny), ey], offsets=[-1, 0, 1],
                      shape=(Ny, Ny), format='csr') / (2*dy)
        Adv_x = sp.kron(Iy, Cx, format='csr')
        Adv_y = sp.kron(Cy, Ix, format='csr')

        # Enforce homogeneous Dirichlet BC by zeroing rows at boundary nodes
        # assume `self.dirichlet` is array of global indices of Dirichlet nodes for both u and v
        # bc_idx = np.concatenate([self.dirichlet, self.dirichlet + A_diff_scalar.shape[0]])
        # A_diff[bc_idx, :] = 0

        # Store operators for use in rhs and C(x)
        self.A_diff = A_diff
        self.Adv_x = Adv_x
        self.Adv_y = Adv_y

    def Cadv(self, x):
        """
        Build the nonlinear convection matrix C(x) = [u; v] · ∇.
        """
        n = self.A_diff.shape[0] // 2

        # split velocity vector
        u = x[:n]
        v = x[n:]

        # elementwise scale advective stencils
        Ux = sp.diags(u, offsets=0, format='csr') @ self.Adv_x
        Vy = sp.diags(v, offsets=0, format='csr') @ self.Adv_y
        conv = Ux + Vy

        # expand to block for [u; v]
        C = sp.block_diag([conv, conv], format='csr')

        return C

    def jacobian(self, t, x):
        """
        Assemble the Jacobian J = d/dx [A_diff*x - C(x)x] = -C(x) - dC(x)x + A_diff.
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

        # diagonal contributions
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

    # TO DO: avoid recomputing the source term in IMEX-RB?
    # @cached_property
    def source_term(self, t):
        # Create array with shape (self.soldim, *self.shape)
        F = np.zeros((self.soldim, *self.shape))
        interior = (slice(1, -1), slice(1, -1))
        # Get source values for the interior points
        source_values = self._f(t, *[mesh[interior] for mesh in self.coords])
        # Assign source values to interior of F
        for i in range(self.soldim):
            F[i][interior] = source_values[i]
        return F.reshape(self.soldim, -1).flatten()

    def rhs(self, t, x):
        """
        Return the evaluation of RHS of the Cauchy problem.
        """
        return self.A_diff @ x - self.Cadv(x) @ x + self.source_term(t)

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
