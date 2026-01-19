import numpy as np
from grad_verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Frank-Wolfe Homotopy Solver for Column Subset Selection.
    Includes Tikhonov Regularization and Numerical Stability Clips.
    """

    def __init__(self, A, k, alpha=0.01, n_steps=500, n_mc_samples=50):
        self.raw_p = A.shape[0]
        self.k = k
        self.alpha = alpha
        self.n = n_steps
        self.n_mc = n_mc_samples

        # --- FIX 1: Tikhonov Regularization ---
        # Prevents singularity crashes.
        A_reg = A + 1e-6 * np.eye(self.raw_p)

        # --- FIX 2: Spectral Normalization ---
        # Scales largest eigenvalue to 1.0 for consistent hyperparams.
        spectral_norm = np.linalg.norm(A_reg, 2)
        self.A = A_reg / spectral_norm
        self.p = self.raw_p
        self.scale_factor = spectral_norm

        # --- Eigenvalue Calculation ---
        evals = np.linalg.eigvalsh(self.A)
        self.eta_p = max(evals[0], 1e-9)
        self.eta_1 = evals[-1]

    def _get_lmo_solution(self, grad):
        idx = np.argpartition(grad, -self.k)[-self.k :]
        s = np.zeros(self.p)
        s[idx] = 1.0
        return s

    def _check_kkt(self, current_s, grad):
        idx_selected = np.where(current_s > 0.5)[0]
        idx_unselected = np.where(current_s <= 0.5)[0]

        if len(idx_selected) == 0 or len(idx_unselected) == 0:
            return True

        min_selected = np.min(grad[idx_selected])
        max_unselected = np.max(grad[idx_unselected])

        return min_selected >= max_unselected - 1e-6

    def solve(self, verbose=True):
        # 1. Initialization
        epsilon = 0.1 * (self.k / self.p)

        # Theoretical delta
        theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)

        # --- FIX 3: Engineering Safety Floor ---
        # Ensures gradient signal is strong enough to escape local optima.
        delta_0 = max(theoretical_delta, 0.001)

        # Decay/Growth Rate (Targeting eta_1)
        if self.n > 1:
            if delta_0 < self.eta_1:
                r = (self.eta_1 / delta_0) ** (1 / (self.n - 1))
            else:
                r = 1.0
        else:
            r = 1.0

        # Start unbiased
        t = np.full(self.p, self.k / self.p)
        curr_delta = delta_0

        if verbose:
            print(f"Starting FW-Homotopy: p={self.p}, k={self.k}")

        # 2. Main Loop
        for l in range(1, self.n + 1):
            curr_delta = delta_0 * (r ** (l - 1))

            # Stochastic Gradient Estimation
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc)
            ]

            grad = BooleanRelaxation.grad_g_analytical(
                self.p, curr_delta, t, self.A, xi_samples
            )

            # Frank-Wolfe Step
            s = self._get_lmo_solution(grad)
            t = (1 - self.alpha) * t + self.alpha * s

            # --- FIX 4: Solution Clipping (New!) ---
            # Keep t away from 0.0 and 1.0 to prevent numerical explosion
            # in gradient calculations (div by zero).
            t = np.clip(t, 0.001, 0.999)

            # 3. Optimality Check (Early Exit)
            dist_to_boundary = np.minimum(t, 1 - t)

            if np.min(dist_to_boundary) <= 1e-4:
                grad_at_s = BooleanRelaxation.grad_g_analytical(
                    self.p, curr_delta, s, self.A, xi_samples
                )
                if self._check_kkt(s, grad_at_s):
                    if verbose:
                        print(f"  [Step {l}/{self.n}] Converged Early!")
                    return s

        # 4. Final Rounding
        final_s = self._get_lmo_solution(t)
        return final_s
