import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Frank-Wolfe Homotopy Solver for Column Subset Selection.
    Corrected for MINIMIZATION of the A-Optimality objective.
    """

    def __init__(self, A, k, alpha=0.01, n_steps=500, n_mc_samples=50):
        self.raw_p = A.shape[0]
        self.k = k
        self.alpha = alpha
        self.n = n_steps
        self.n_mc = n_mc_samples

        # Tikhonov Regularization (Safety against singularity)
        A_reg = A + 1e-6 * np.eye(self.raw_p)

        self.A = A_reg
        self.p = self.raw_p

        # Eigenvalue Calculation
        evals = np.linalg.eigvalsh(self.A)
        self.eta_p = max(evals[0], 1e-9)
        self.eta_1 = evals[-1]

    def _get_lmo_solution(self, grad):
        """
        LMO for MINIMIZATION:
        Select indices corresponding to the k SMALLEST gradients.
        """
        # argpartition moves the k smallest elements to the front
        idx = np.argpartition(grad, self.k)[: self.k]

        s = np.zeros(self.p)
        s[idx] = 1.0
        return s

    def _check_kkt(self, current_s, grad):
        """
        KKT Optimality Check for MINIMIZATION.
        max(grad[selected]) <= min(grad[unselected])
        """
        idx_selected = np.where(current_s > 0.5)[0]
        idx_unselected = np.where(current_s <= 0.5)[0]

        if len(idx_selected) == 0 or len(idx_unselected) == 0:
            return True

        max_selected = np.max(grad[idx_selected])
        min_unselected = np.min(grad[idx_unselected])

        return max_selected <= min_unselected + 1e-6

    def solve(self, n_restarts=1, verbose=True):
        """
        Solves the problem using FW-Homotopy with Random Restarts.
        """
        best_val = np.inf
        best_s = None

        for run_i in range(n_restarts):
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

            t = np.full(self.p, self.k / self.p)
            curr_delta = delta_0

            # --- 2. Optimization Loop ---
            for l in range(1, self.n + 1):
                curr_delta = delta_0 * (r ** (l - 1))

                xi_samples = [
                    np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
                ]

                # Compute Gradient
                # Using grad_z_analytical (gradient of z = -g) for minimization
                grad = BooleanRelaxation.grad_z_analytical(
                    self.p, curr_delta, t, self.A, xi_samples
                )

                # LMO: Pick smallest gradients
                s = self._get_lmo_solution(grad)

                # Update t
                t = (1 - self.alpha) * t + self.alpha * s
                t = np.clip(t, 0.001, 0.999)

                # Early Exit Check
                dist_to_boundary = np.minimum(t, 1 - t)
                if np.min(dist_to_boundary) <= 1e-4:
                    curr_delta = self.eta_1
                    grad_at_s = BooleanRelaxation.grad_z_analytical(
                        self.p, curr_delta, s, self.A, xi_samples
                    )
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
            try:
                A_sub = self.A[np.ix_(idx, idx)]
                val = np.trace(np.linalg.inv(A_sub))
            except np.linalg.LinAlgError:
                val = np.inf

            # Keep the best result across restarts
            if val < best_val:
                best_val = val
                best_s = final_s_run

        return best_s
