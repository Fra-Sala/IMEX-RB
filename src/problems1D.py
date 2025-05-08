import numpy as np
import scipy


class PDE1DBase:
    """
    Base class for 1D PDEs with Dirichlet BCs.
    Subclasses must implement `assemble_stencil()` to build the operator L
    and `source_term(t)` to return forcing vector of length N.
    """
    def __init__(self, N, L, kappa, bc_left=0.0, bc_right=0.0):
        self.N = N
        self.L = L
        self.kappa = kappa
        self.dx = L/(N-1)
        # Dirichlet BC values or callables g_L(t), g_R(t)
        self.bc_left = bc_left if callable(bc_left) else (lambda t: bc_left)
        self.bc_right = bc_right if callable(bc_right) else (lambda t: bc_right)
        # build Laplacian stencil operator L (no BC stamping)
        self.A = None
        self.assemble_stencil()

    def assemble_stencil(self):
        """
        Build the finite-difference Laplacian (kappa*D2) as sparse matrix,
        but do NOT enforce BC rows here.
        """
        raise NotImplementedError

    def source_term(self, t):
        """
        Return forcing vector f(t) of length N (zero at BC entries).
        """
        raise NotImplementedError
    
    def enforce_bcs(self, v, t):
        v[0] = self.bc_left(t)
        v[-1] = self.bc_right(t)
        return v

    def apply_bc(self, M, b, t):
        """
        Apply Dirichlet BCs by lifting:
        - Build uD = [g_L(t), 0,...,0, g_R(t)]^T
        - Modify rhs: b <- b - M@uD
        - Stamp rows: set rows of M to identity rows, and b[i]=uD[i]
        Returns M_bc, b_bc
        """
        # Dirichlet vector
        uD = np.zeros(self.N)
        uD = self.enforce_bcs(uD, t)
        # lift rhs
        b_lifted = b - M @ uD
        # stamp rows in M
        M_bc = M.tolil()
        # Change Dirichlet rows to identity
        for i in (0, self.N-1):
            M_bc[i, :] = 0
            M_bc[i, i] = 1.0
            b_lifted[i] = uD[i]
        return M_bc.tocsr(), b_lifted

    def initial_condition(self, t0=0.0):
        """
        Default zero initial condition (or subclasses override).
        """
        return np.zeros(self.N)


# 1D Heat equation
def Heat1D(N, L, kappa, bc_left=0.0, bc_right=0.0, f=None):
    class Heat(PDE1DBase):
        def __init__(self):
            super().__init__(N, L, kappa, bc_left, bc_right)
            
            self._f = f if f is not None else (lambda t, x: 0.0)
            # spatial grid
            self.domain = np.linspace(0, L, N)

        def assemble_stencil(self):
            # build kappa * second derivative
            coef = self.kappa / self.dx**2
            diags = [coef*np.ones(self.N-1),
                     -2*coef*np.ones(self.N),
                     coef*np.ones(self.N-1)]
            self.A = scipy.sparse.diags(diags, (-1, 0, 1),
                                        shape=(self.N, self.N),
                                        format='csr')

        def source_term(self, t):
            f_vec = np.zeros(self.N)
            f_vec[1:-1] = self._f(t, self.domain[1:-1])
            return f_vec

        def exact_solution(self, x, t, lam=np.pi):
            return np.sin(lam*x)*np.exp(-self.kappa*lam**2*t)
        
        def initial_condition(self, t0=0):
            return self.exact_solution(self.domain, t0)

    return Heat()
