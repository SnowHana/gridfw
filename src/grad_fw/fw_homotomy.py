import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Implements the Frank-Wolfe Homotopy Algorithm (Stochastic Version).
    Automatically generates fresh random vectors at every step for better convergence.
    """

    def __init__(self, A, k, alpha=0.1, n_steps=100, n_mc_samples=50):
        """
        Args:
            A (np.ndarray): The symmetric positive definite matrix (X^T X).
            k (int): Cardinality constraint.
            alpha (float): Step size for Frank-Wolfe update.
            n_steps (int): Total number of homotopy iterations.
            n_mc_samples (int): Monte Carlo samples for gradient expectation.
        """
        self.A = A
        self.p = A.shape[0]
        self.k = k
        self.alpha = alpha
        self.n = n_steps
        self.n_mc = n_mc_samples

        # Pre-compute eigenvalues
        eigenvals = np.linalg.eigvalsh(A)
        # Safety floor for eigenvalues to prevent math errors on ill-conditioned data
        self.eta_p = max(eigenvals[0], 1e-6)
        self.eta_1 = max(eigenvals[-1], 1.0)

    def _get_lmo_solution(self, grad):
        """Linear Maximization Oracle: Selects top-k indices."""
        top_k_indices = np.argsort(grad)[-self.k :]
        s = np.zeros(self.p)
        s[top_k_indices] = 1.0
        return s

    def _check_kkt(self, s, grad):
        """Checks if the solution satisfies optimality conditions."""
        idx_1 = np.where(s > 0.5)[0]
        idx_0 = np.where(s < 0.5)[0]
        if len(idx_1) == 0 or len(idx_0) == 0:
            return True
        return np.min(grad[idx_1]) >= np.max(grad[idx_0])

    def solve(self, verbose=True):
        # 1. Initialization
        epsilon = 0.1 * (self.k / self.p)
        theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)
        delta_0 = max(theoretical_delta, 0.1)

        # Calculate decay rate r
        if self.n > 1:
            if delta_0 < self.eta_1:
                r = (self.eta_1 / delta_0) ** (1 / (self.n - 1))
            else:
                r = 1.0
        else:
            r = 1.0

        t = np.full(self.p, self.k / self.p)
        curr_delta = delta_0

        if verbose:
            print(f"Starting FW-Homotopy [Stochastic]: p={self.p}, k={self.k}")

        # 2. Main Loop
        for l in range(1, self.n + 1):
            curr_delta = delta_0 * (r ** (l - 1))

            # ALWAYS generate fresh samples (The "Smart" Way)
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
            ]

            grad = BooleanRelaxation.grad_g_analytical(
                self.p, curr_delta, t, self.A, xi_samples
            )

            s = self._get_lmo_solution(grad)
            t = (1 - self.alpha) * t + self.alpha * s

            # KKT Check
            dist_to_boundary = np.minimum(t, 1 - t)
            if np.min(dist_to_boundary) <= 1e-4:
                # Use current samples for KKT check
                grad_at_s = BooleanRelaxation.grad_g_analytical(
                    self.p, curr_delta, s, self.A, xi_samples
                )
                if self._check_kkt(s, grad_at_s):
                    if verbose:
                        print(f"  [Step {l}] KKT Optimality Certified!")
                    return s

        final_s = self._get_lmo_solution(t)
        return final_s
