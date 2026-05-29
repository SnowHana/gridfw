import numpy as np
from grad_fw.verif.core import BooleanRelaxation


class FWHomotopySolver:
    """
    Frank-Wolfe Homotopy Solver for Column Subset Selection.
    Corrected for MINIMIZATION of the A-Optimality objective.
    # TODO: Alpha : Adaptive alpha or standard frank wolfe alpha might be nice.
    """

    def __init__(
        self, A, k, alpha=0.01, n_steps=None, n_mc_samples=None, objective_type="cssp"
    ):
        self.raw_p = A.shape[0]
        self.k = k
        self.alpha = alpha
        self.n_steps = n_steps
        self.objective_type = objective_type

        # Adaptive Sampling Strategy
        # Allow at least 20 steps per features
        if n_steps is None:
            self.n_steps = max(1000, int(20 * self.k))
        else:
            self.n_steps = n_steps

        if n_mc_samples is None:
            self.n_mc_samples = 50
        else:
            self.n_mc_samples = n_mc_samples

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

    def solve(self, verbose=True):
        """
        Solve CSSP using FW-Homotopy

        :param verbose: Print solver configuration before running
        """
        if verbose:
            print(
                f"[FWHomotopy] p={self.p}, k={self.k}, steps={self.n_steps}, "
                f"n_mc={self.n_mc_samples}, alpha={self.alpha}"
            )

        # --- 1. Initialization ---
        epsilon = 0.1 * (self.k / self.p)
        theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)

        # Safety Floor: Prevents "wandering" in flat regions
        min_scale = 1e-3 * self.eta_1
        # NOTE: Clipping: We don't want delta_0 to be too small
        # Choose max(theory, 1e-6 (practically 0), eta_1 / 1000)
        delta_0 = max(theoretical_delta, 1e-6, min_scale)

        # Decay Rate : delta_1 ... delta_n = eta_1
        if self.n_steps > 1:
            r = (self.eta_1 / delta_0) ** (1 / (self.n_steps - 1))
        else:
            r = 1.0

        t = np.full(self.p, self.k / self.p)  # O(p)

        # --- 2. Main Optimization Loop ---
        for l in range(1, self.n_steps + 1):  # O(n)
            curr_delta = delta_0 * (r ** (l - 1))

            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc_samples)
            ]  # O(n_mc * p)

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

        # Overall timecomplexity: O(n * p^3 + n * p^2 * n_mc)
        return s

    def solve_with_history(self, record_every=10, verbose=True):
        """Same as solve() but also returns snapshots of (t, s, Delta) for animation.

        Returns (s, history) where history is a list of dicts recorded every
        `record_every` steps plus the final step.
        """
        if verbose:
            print(
                f"[FWHomotopy] p={self.p}, k={self.k}, steps={self.n_steps}, "
                f"n_mc={self.n_mc_samples}, alpha={self.alpha}"
            )

        epsilon = 0.1 * (self.k / self.p)
        theoretical_delta = (3 * self.eta_p * epsilon**2) / (1 + 3 * epsilon**2)
        min_scale = 1e-3 * self.eta_1
        delta_0 = max(theoretical_delta, 1e-6, min_scale)
        r = (
            (self.eta_1 / delta_0) ** (1 / (self.n_steps - 1))
            if self.n_steps > 1
            else 1.0
        )

        t = np.full(self.p, self.k / self.p)
        history = []

        for l in range(1, self.n_steps + 1):
            curr_delta = delta_0 * (r ** (l - 1))
            xi_samples = [
                np.random.choice([-1, 1], size=self.p) for _ in range(self.n_mc_samples)
            ]
            grad = BooleanRelaxation.grad_z_analytical(
                self.p, curr_delta, t, self.A, xi_samples
            )
            s = self._get_lmo_solution(grad)
            t = (1 - self.alpha) * t + self.alpha * s
            t = np.clip(t, 0.001, 0.999)

            if l % record_every == 0 or l == self.n_steps:
                history.append(
                    {"step": l, "delta": curr_delta, "t": t.copy(), "s": s.copy()}
                )

        return s, history
