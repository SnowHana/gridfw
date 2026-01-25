import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Frank-Wolfe Homotopy Solver for Column Subset Selection.
    Corrected for MINIMIZATION of the A-Optimality objective.
    """

    def __init__(
        self, A, k, alpha=0.01, n_steps=500, n_mc_samples=50, objective_type="cssp"
    ):
        self.raw_p = A.shape[0]
        self.k = k
        self.alpha = alpha
        self.n = n_steps
        self.objective_type = objective_type

        # Adaptive Sampling Strategy
        # For small k, we need more samples to reduce variance
        if self.k < 20 and n_mc_samples == 50:
            self.n_mc = 300
        else:
            self.n_mc = n_mc_samples

        # Tikhonov Regularization : A = A + lambda * I > 0 (Pos. Def.)
        A_reg = A + 1e-6 * np.eye(self.raw_p)

        self.A = A_reg
        self.p = self.raw_p

        # Eigenvalue Calculation
        evals = np.linalg.eigvalsh(self.A)  # O(p^3)
        self.eta_p = max(evals[0], 1e-9)
        self.eta_1 = evals[-1]

        # Precompute A^2 for CSSP objective evaluation
        self.A2 = self.A @ self.A  # O(p^2)

        # Init. takes O(p^3)

    def _get_lmo_solution(self, grad):
        """
        LMO for MINIMIZATION:
        Select indices corresponding to the k SMALLEST gradients.
        """
        # argpartition moves the k smallest elements to the front
        if self.k == self.p:
            return np.ones(self.p)

        idx = np.argpartition(grad, self.k)[: self.k]  # O(p)

        s = np.zeros(self.p)
        s[idx] = 1.0
        return s

    def _check_kkt(self, current_s, grad):
        """
        KKT Optimality Check for MINIMIZATION.
        max(grad[selected]) <= min(grad[unselected])
        """
        idx_selected = np.where(current_s > 0.5)[0]
        idx_unselected = np.where(current_s <= 0.5)[0]  # O(p)

        if len(idx_selected) == 0 or len(idx_unselected) == 0:
            return True

        max_selected = np.max(grad[idx_selected])  # O(k)
        min_unselected = np.min(grad[idx_unselected])  # O(p-k)

        return max_selected <= min_unselected + 1e-6

    def solve(self, n_restarts=1, verbose=True):
        """
        Solves the problem using FW-Homotopy with Random Restarts.
        """
        best_val = -np.inf
        best_s = None

        for run_i in range(n_restarts):  # O(n_restarts)
            if verbose and n_restarts > 1:
                print(f"--- Restart {run_i+1}/{n_restarts} ---")

            # --- 1. Initialization ---
            epsilon = 0.1 * (self.k / self.p)
            theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)

            # Safety Floor: Prevents "wandering" in flat regions
            delta_0 = max(theoretical_delta, 0.001)

            # Decay Rate
            if self.n > 1:
                if delta_0 < self.eta_1:
                    r = (self.eta_1 / delta_0) ** (1 / (self.n - 1))
                else:
                    r = 1.0
            else:
                r = 1.0

            t = np.full(self.p, self.k / self.p)  # O(p)
            curr_delta = delta_0

            # --- Sample Generation (SAA) ---
            # Create m Rademacher vectors at the start of the run (per restart)
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
            ]  # O(n_mc * p)

            # --- 2. Optimization Loop ---
            for l in range(1, self.n + 1):  # O(n)
                curr_delta = delta_0 * (r ** (l - 1))

                # Compute Gradient
                if self.objective_type == "portfolio":
                    # Portfolio: Maximize 1^T A_S^-1 1
                    grad = BooleanRelaxation.grad_portfolio_analytical(
                        self.p, curr_delta, t, self.A
                    )
                else:
                    # CSSP: Maximize Tr(X^T P_S X)
                    # Using grad_z_analytical (gradient of z = -g) for minimization
                    grad = BooleanRelaxation.grad_z_analytical(
                        self.p, curr_delta, t, self.A, xi_samples
                    )

                # LMO: Pick smallest gradients
                s = self._get_lmo_solution(grad)  # O(p)

                # Update t
                t = (1 - self.alpha) * t + self.alpha * s
                t = np.clip(t, 0.001, 0.999)

                # Early Exit Check
                dist_to_boundary = np.minimum(t, 1 - t)
                if np.min(dist_to_boundary) <= 1e-4:
                    curr_delta = self.eta_1
                    if self.objective_type == "portfolio":
                        grad_at_s = BooleanRelaxation.grad_portfolio_analytical(
                            self.p, curr_delta, s, self.A
                        )
                    else:
                        grad_at_s = BooleanRelaxation.grad_z_analytical(
                            self.p, curr_delta, s, self.A, xi_samples
                        )  # O(p^3 + p^2 * n_mc)
                    if self._check_kkt(s, grad_at_s):
                        if verbose:
                            print(f"  [Step {l}/{self.n}] Converged Early!")
                        # Break inner loop, use current 's' as result for this run
                        break

            # --- 3. Final Evaluation for this Restart ---
            # We want to pick indices with LARGEST t, so we pass -t to LMO (which picks smallest)
            final_s_run = self._get_lmo_solution(-t)

            # Calculate objective to compare restarts
            idx = np.where(final_s_run > 0.5)[0]
            if len(idx) == 0:
                val = -np.inf
            else:
                try:
                    A_sub = self.A[np.ix_(idx, idx)]
                    inv_A_sub = np.linalg.pinv(A_sub)  # O(p^3)
                    if self.objective_type == "portfolio":
                        # Maximize 1^T A_S^-1 1
                        val = np.sum(inv_A_sub)
                    else:
                        # Maximize Tr( A_S^-1 (A^2)_S )
                        A2_sub = self.A2[np.ix_(idx, idx)]
                        val = np.trace(inv_A_sub @ A2_sub)
                except np.linalg.LinAlgError:
                    val = -np.inf

            # Keep the best result across restarts (MAXIMIZE)
            if val > best_val or best_s is None:
                best_val = val
                best_s = final_s_run
        # Overall timecomplexity: O(n_restarts * n * p^3 + n_restarts * n * p^2 * n_mc)
        return best_s
