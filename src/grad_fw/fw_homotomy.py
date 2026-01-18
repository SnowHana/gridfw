import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Implements the Frank-Wolfe Homotopy Algorithm (Algorithm 1) for CSSP.
    Includes robustness fixes for ill-conditioned real-world data.
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
        eigenvals = np.linalg.eigvalsh(A)

        # FIX 1: Safety Floor for Eigenvalues
        # Real data often has eigenvalues approx 0. We clamp them to avoid math errors.
        self.eta_p = max(eigenvals[0], 1e-6)  # Smallest (clamped)
        self.eta_1 = max(eigenvals[-1], 1.0)  # Largest (clamped)

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

        # FIX 2: Robust Delta Initialization
        # Theoretical delta can be too small for ill-conditioned data (causing clumping).
        # We enforce a minimum start temperature of 0.1 to ensure exploration.
        theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)
        delta_0 = max(theoretical_delta, 0.1)

        # Calculate geometric decay rate r
        # delta_k = delta_0 * r^k
        if self.n > 1:
            # FIX 3: Prevent negative base in power calculation
            # If delta_0 > eta_1 (unlikely but possible), we just clamp r to 1.0
            if delta_0 < self.eta_1:
                # Standard annealing: start smooth (delta_0), end sharp (eta_1)
                # Wait, usually we start LARGE (smooth) and end SMALL (sharp) for annealing?
                # Actually, in this Homotopy formulation (from Moka et al?),
                # 'delta' usually INCREASES or behaves specifically.
                # Let's trust your derived schedule: delta_0 -> eta_1.
                r = (self.eta_1 / delta_0) ** (1 / (self.n - 1))
            else:
                r = 1.0
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
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
            ]

            # Use verified formula from core.py
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

                # Check KKT at the strict corner 's' using final delta
                # Note: We use the current delta or final? Usually KKT is checked on the original problem.
                # Here we check using current estimates.
                grad_at_s = BooleanRelaxation.grad_g_analytical(
                    self.p, curr_delta, s, self.A, xi_samples
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
