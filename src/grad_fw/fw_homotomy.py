import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Implements the Frank-Wolfe Homotopy Algorithm (Algorithm 1) for CSSP.
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

        # Pre-compute eigenvalues for the schedule
        # A is Symmetric PD.
        eigenvals = np.linalg.eigvalsh(A)
        self.eta_p = eigenvals[0]  # Smallest
        self.eta_1 = eigenvals[-1]  # Largest

    def _get_lmo_solution(self, grad):
        """
        Linear Maximization Oracle (LMO).
        Finds binary vector s that MAXIMIZES <s, grad> subject to |s| <= k.
        """
        # For Maximization: Select indices of the k LARGEST gradient components.
        top_k_indices = np.argsort(grad)[-self.k :]

        s = np.zeros(self.p)
        s[top_k_indices] = 1.0
        return s

    def _check_kkt(self, s, grad):
        """
        Checks KKT optimality condition for Maximization.
        Condition: The smallest gain we KEPT >= The largest gain we DISCARDED.
        """
        idx_1 = np.where(s > 0.5)[0]  # Selected
        idx_0 = np.where(s < 0.5)[0]  # Unselected

        if len(idx_1) == 0 or len(idx_0) == 0:
            return True

        min_g_selected = np.min(grad[idx_1])
        max_g_unselected = np.max(grad[idx_0])

        return min_g_selected >= max_g_unselected

    def solve(self, verbose=True):
        """Executes the FW-Homotopy algorithm."""

        # 1. Initialization Parameters
        tau = 1e-4
        epsilon = 0.1 * (self.k / self.p)

        # Calculate start delta and geometric rate r
        delta_0 = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)
        if self.n > 1:
            r = (self.eta_1 / delta_0) ** (1 / (self.n - 1))
        else:
            r = 1.0

        # Start at center of the polytope
        t = np.full(self.p, self.k / self.p)
        curr_delta = delta_0

        if verbose:
            print(f"Starting FW-Homotopy (Maximization): p={self.p}, k={self.k}")
            print(
                f"Schedule: delta_0={delta_0:.2e} -> eta_1={self.eta_1:.2e} (Steps={self.n})"
            )

        # 2. Main Homotopy Loop
        final_s = np.zeros(self.p)

        for l in range(1, self.n + 1):
            # A. Update delta
            curr_delta = delta_0 * (r ** (l - 1))

            # B. Compute Stochastic Gradient
            # Generate fresh Monte Carlo samples
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
            ]

            # Use your VERIFIED formula from core.py
            grad = BooleanRelaxation.grad_g_analytical(
                self.p, curr_delta, t, self.A, xi_samples
            )

            # C. Linear Oracle (Finding direction s)
            s = self._get_lmo_solution(grad)

            # D. Frank-Wolfe Update
            t = (1 - self.alpha) * t + self.alpha * s

            # E. Boundary / KKT Check
            dist_to_boundary = np.minimum(t, 1 - t)

            # If we hit the boundary (binary solution found)
            if np.min(dist_to_boundary) <= tau:
                if verbose:
                    print(f"  [Step {l}] Boundary reached. Checking KKT...")

                # Check KKT at the strict corner 's' using final delta = eta_1
                grad_at_s = BooleanRelaxation.grad_g_analytical(
                    self.p, self.eta_1, s, self.A, xi_samples
                )

                if self._check_kkt(s, grad_at_s):
                    if verbose:
                        print(f"  SUCCESS: KKT Optimality Certified!")
                    return s
                else:
                    if verbose:
                        print(f"  ... KKT failed. Continuing.")

        # 3. Final Rounding (if loop finishes without KKT break)
        if verbose:
            print("  Loop finished. Returning rounded solution.")
        final_s = self._get_lmo_solution(t)  # Deterministic rounding of t
        return final_s
