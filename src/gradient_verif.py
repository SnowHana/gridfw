import numpy as np


class GradientVerifier:
    def __init__(self, h=1e-7, rel_threshold=1e-5):
        """
        Initializes the verifier with numerical precision settings.
        Args:
            h (float): Step size for finite difference.
            rel_threshold (float): Acceptable RELATIVE error threshold.
        """
        self.h = h
        self.threshold = rel_threshold

    def _generate_problem(self, p):
        """Generates a random problem instance (A, t, xi, b)."""
        X = np.random.randn(p, p)
        A = X.T @ X  # Positive definite matrix A = X^T X

        # Changed to 0.01-0.99 to test wider range while avoiding 0/1 singularities
        t = np.random.uniform(0.01, 0.99, size=p)

        xi = np.random.choice([-1, 1], size=p)  # Rademacher vector [cite: 65]
        b = A @ xi  # b defined as X^T X xi [cite: 56]
        return A, t, b, xi

    def _get_eigenvalues(self, A):
        """Returns eta_p (smallest) and eta_1 (largest) eigenvalues."""
        eigenvals = np.linalg.eigvalsh(A)
        return eigenvals[0], eigenvals[-1]

    def _calculate_gradient_error(self, p, delta, t, b, A):
        """
        Computes Relative Difference between analytical and numerical gradients.
        Analytical Formula: 2 * delta * (Pi_inv @ b)^2 / t^3
        """

        # Helper for function value f_delta
        def get_f_val(t_vec):
            # NOTE : This calculates Dt for EVERY POINT
            # Because Dt actually changes relative to t
            Dt = np.diag(1.0 / (t_vec**2)) - np.eye(p)
            Pi_t = A + delta * Dt
            # Matrix inverse must be recomputed for every t
            return b.T @ np.linalg.inv(Pi_t) @ b

        # [cite_start]1. Analytical Gradient [cite: 148, 225]
        Pi_t = A + delta * (np.diag(1.0 / (t**2)) - np.eye(p))
        Pi_inv = np.linalg.inv(Pi_t)
        # Element-wise squaring and division by t^3
        grad_analytical = 2 * delta * ((Pi_inv @ b) ** 2) / (t**3)

        # 2. Numerical Gradient (Central Difference)
        grad_numerical = np.zeros(p)
        for i in range(p):
            t_plus = t.copy()
            t_plus[i] += self.h
            t_minus = t.copy()
            t_minus[i] -= self.h
            grad_numerical[i] = (get_f_val(t_plus) - get_f_val(t_minus)) / (2 * self.h)

        # 3. Error Metrics
        abs_error = np.linalg.norm(grad_analytical - grad_numerical)
        grad_norm = np.linalg.norm(grad_analytical)

        # Avoid division by zero
        rel_error = abs_error / (grad_norm + 1e-10)

        # If the gradient itself is extremely small (e.g., < 1e-8), for small delta cases
        if grad_norm < 1e-8:
            if abs_error < 1e-8:
                rel_error = 0.0

        return rel_error, abs_error, grad_norm

    def _print_stats(self, results, title):
        """Standardized statistical reporting for (Rel, Abs, Norm) tuples."""
        # Unpack results
        rel_errors = [r[0] for r in results]
        grad_norms = [r[2] for r in results]

        print(f"\n--- {title} ---")
        print(f"Count:        {len(results)}")
        print(f"Mean Rel Err: {np.mean(rel_errors):.2e}")
        print(f"Median Rel:   {np.median(rel_errors):.2e}")
        print(f"Max Rel Err:  {np.max(rel_errors):.2e}")
        print(
            f"Max Grad Mag: {np.max(grad_norms):.2e}"
        )  # Check if gradients are exploding

        if np.max(rel_errors) > self.threshold:
            print(f"WARNING: Max relative error exceeds threshold ({self.threshold})")
        else:
            print("STATUS: PASSED")

    # --- User Facing Test Functions ---

    def test_general_random(self, p, num_matrices, points_per_matrix):
        """Test 1: Random Deltas and Points (General Robustness)."""
        results = []
        for _ in range(num_matrices):
            A, _, _, _ = self._generate_problem(p)
            for _ in range(points_per_matrix):
                # Random delta in reasonable range
                delta = np.random.uniform(0.1, 10.0)
                # Regenerate t and b for variety
                t = np.random.uniform(0.01, 0.99, size=p)
                xi = np.random.choice([-1, 1], size=p)
                b = A @ xi

                err_tuple = self._calculate_gradient_error(p, delta, t, b, A)
                results.append(err_tuple)

        self._print_stats(results, "Test 1: General Random Cases")

    def test_algorithm_path(self, p, num_matrices, n_steps=10, k=None):
        """
        Test 2: Simulates the Homotopy Algorithm's delta schedule.
        [cite_start]Schedule: delta_0 -> ... -> eta_1 [cite: 205]
        """
        if k is None:
            k = max(1, p // 3)  # Default k if not provided
        results = []

        print(f"\n--- Test 2: Algorithm Path (Homotopy) ---")
        print(f"Simulating {num_matrices} matrices with {n_steps} homotopy steps each.")

        for _ in range(num_matrices):
            A, t, b, xi = self._generate_problem(p)
            eta_p, eta_1 = self._get_eigenvalues(A)

            # [cite_start]Algorithm parameters [cite: 205]
            epsilon = 0.1 * (k / p)
            delta_0 = (3 * eta_p * epsilon**2) / (1 + 3 * epsilon**2)

            # Calculate geometric rate r
            if n_steps > 1:
                r = (eta_1 / delta_0) ** (1 / (n_steps - 1))
            else:
                r = 1

            # Iterate through the schedule
            current_delta = delta_0
            for step in range(n_steps):
                err_tuple = self._calculate_gradient_error(p, current_delta, t, b, A)
                results.append(err_tuple)
                current_delta *= r  # Update delta

        self._print_stats(results, "Homotopy Path Statistics")

    def test_convexity_boundary(self, p, num_matrices):
        """
        Test 3: Checks gradient stability around eta_1.
        [cite_start]eta_1 is where f_delta becomes strictly concave[cite: 175, 217].
        """
        results = []
        for _ in range(num_matrices):
            A, t, b, xi = self._generate_problem(p)
            _, eta_1 = self._get_eigenvalues(A)

            # Test specifically at boundaries
            test_deltas = [
                eta_1 * 0.9,  # Just below (Transition)
                eta_1,  # Exact Boundary
                eta_1 * 1.1,  # Just above (Strictly Concave)
            ]

            for d in test_deltas:
                err_tuple = self._calculate_gradient_error(p, d, t, b, A)
                results.append(err_tuple)

        self._print_stats(results, "Test 3: Convexity Boundary (Around eta_1)")


# --- Main Execution ---
def run_interactive_suite():
    print("=== Boolean Relaxation Gradient Verification Suite ===")
    try:
        p = int(input("Enter dimension of matrix (p): "))
        n_mat = int(input("Enter number of random matrices to test: "))
        n_pts = int(input("Enter number of points per matrix: "))
    except ValueError:
        print("Invalid input. Using defaults: p=5, matrices=3, points=3")
        p, n_mat, n_pts = 5, 3, 3

    verifier = GradientVerifier()

    # 1. Run General Random Tests
    verifier.test_general_random(p, n_mat, n_pts)

    # 2. Run Algorithm Path Simulation (Simulating n_pts steps)
    verifier.test_algorithm_path(p, n_mat, n_steps=n_pts)

    # 3. Run Boundary Tests
    verifier.test_convexity_boundary(p, n_mat)


if __name__ == "__main__":
    run_interactive_suite()
