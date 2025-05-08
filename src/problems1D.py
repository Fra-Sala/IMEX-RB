import numpy as np
import scipy


class PDE1DBase:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.dx = L/(N-1)
        self.A = None

    def assemble_matrix(self):
        raise NotImplementedError

    def b(self, t):
        raise NotImplementedError
    
    def rhs(self, t, y):
        return self.A.dot(y) + self.b(t)
    
    def exact_sol(self, x, t):
        raise NotImplementedError
    
    def initial_condition(self, t0=0.0):
        raise NotImplementedError


class Heat1D(PDE1DBase):
    def __init__(self, N, L, kappa, bc_left=0.0, bc_right=0.0, f=None):
        super().__init__(N, L)
        self.kappa = kappa
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.f = f if f is not None else (lambda t, x: 0.0)
        self.domain = np.linspace(0, self.L, N)
        self.assemble_matrix()

    def assemble_matrix(self):
        N, dx, k = self.N, self.dx, self.kappa
        coef = k / dx**2
        # tridiagonal stencil
        diags_data = [
            coef * np.ones(N-1),
            -2 * coef * np.ones(N),
            coef * np.ones(N-1),
        ]
        offsets = (-1, 0, 1)
        A = scipy.sparse.diags(diags_data, offsets, shape=(N, N), format='lil')
        # Dirichlet BC rows
        A[0, :] = 0
        A[0, 0] = 1
        A[-1, :] = 0
        A[-1, -1] = 1
        self.A = A.tocsr()

    def b(self, t):
        N, dx, k = self.N, self.dx, self.kappa
        b = np.zeros(N)

        # source term
        b[1:-1] = self.f(t, self.domain[1:-1])

        # enforce BCs (lifting)
        b[1] += k/dx**2 * self.bc_left
        b[-2] += k/dx**2 * self.bc_right
        b[0] = self.bc_left
        b[-1] = self.bc_right
        return b

    def exact_sol(self, x, t, lambda_=np.pi):
        """
        An exact solution for error evaluation, 1D heat equation.
        """
        return np.sin(lambda_*x)*np.exp(-self.kappa*lambda_**2*t)
 
    def initial_condition(self, t0=0.0):
        return self.exact_sol(self.domain, t0)