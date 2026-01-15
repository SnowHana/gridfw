import numpy as np
import pandas as pd
import time


class GradientTestLogger:
    """Handles logging of test results and generation of summary tables."""

    def __init__(self):
        self.logs = []

    def log_entry(
        self, test_name, p, n_samples, max_rel_err, mean_rel_err, max_grad_norm, status
    ):
        self.logs.append(
            {
                "Test Type": test_name,
                "Dim (p)": p,
                "Samples": n_samples,
                "Max Rel Err": f"{max_rel_err:.2e}",
                "Mean Rel Err": f"{mean_rel_err:.2e}",
                "Max Grad": f"{max_grad_norm:.2e}",
                "Status": status,
            }
        )

    def display_summary(self):
        """Prints a pandas DataFrame summary of all tests run."""
        if not self.logs:
            print("No logs to display.")
            return

        df = pd.DataFrame(self.logs)
        print("\n" + "=" * 80)
        print(f"GRADIENT VERIFICATION SUMMARY REPORT")
        print("=" * 80)
        # Adjust column width for readability
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))
        print("=" * 80 + "\n")

    def save_to_csv(self, filename="gradient_verification_log.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Log saved to {filename}")


class GradientVerifier:
    def __init__(self, h=1e-7, rel_threshold=1e-5, logger=None):
        """
        Args:
            h (float): Step size for finite difference.
            rel_threshold (float): Acceptable RELATIVE error threshold.
            logger (GradientTestLogger): Instance to log results.
        """
        self.h = h
        self.threshold = rel_threshold
        self.logger = logger if logger else GradientTestLogger()

    def _generate_problem(self, p):
        X = np.random.randn(p, p)
        A = X.T @ X  # Positive definite [cite: 314, 343]
        t = np.random.uniform(0.01, 0.99, size=p)
        xi = np.random.choice([-1, 1], size=p)  # Rademacher [cite: 311]
        b = A @ xi  # b^(xi) [cite: 315]
        return A, t, b, xi

    def _get_eigenvalues(self, A):
        eigenvals = np.linalg.eigvalsh(A)
        return eigenvals[0], eigenvals[-1]

    def _calculate_gradient_error(self, p, delta, t, b, A):
        """Computes Relative Error between Analytical and Numerical Gradients."""

        # Helper for f_delta
        def get_f_val(t_vec):
            Dt = np.diag(1.0 / (t_vec**2)) - np.eye(p)
            Pi_t = A + delta * Dt
            return b.T @ np.linalg.inv(Pi_t) @ b

        # 1. Analytical Gradient [cite: 407, 484]
        # Formula: 2 * delta * (Pi_inv @ b)^2 / t^3
        Pi_t = A + delta * (np.diag(1.0 / (t**2)) - np.eye(p))
        Pi_inv = np.linalg.inv(Pi_t)
        grad_analytical = 2 * delta * ((Pi_inv @ b) ** 2) / (t**3)

        # 2. Numerical Gradient
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

        # Small Gradient Fix: Trust Analytical if Abs Error is tiny
        if grad_norm < 1e-8 and abs_error < 1e-8:
            rel_error = 0.0

        return rel_error, grad_norm

    def _process_results(self, test_name, p, results):
        """Aggregates results and logs them."""
        rel_errors = [r[0] for r in results]
        grad_norms = [r[1] for r in results]

        max_rel = np.max(rel_errors)
        mean_rel = np.mean(rel_errors)
        max_grad = np.max(grad_norms)

        status = "PASS" if max_rel <= self.threshold else "WARN"

        self.logger.log_entry(
            test_name, p, len(results), max_rel, mean_rel, max_grad, status
        )
        return status

    # --- Test Suites ---

    def test_general_random(self, p, n_matrices, n_points):
        """Stress Test: Random Deltas and Points."""
        results = []
        for _ in range(n_matrices):
            A, _, _, _ = self._generate_problem(p)
            for _ in range(n_points):
                t = np.random.uniform(0.01, 0.99, size=p)
                xi = np.random.choice([-1, 1], size=p)
                b = A @ xi
                delta = np.random.uniform(0.1, 10.0)
                results.append(self._calculate_gradient_error(p, delta, t, b, A))

        return self._process_results("General Random", p, results)

    def test_algorithm_path(self, p, n_matrices, n_steps=10, k=None):
        """Stress Test: Homotopy Path (Algorithm 1 Simulation)."""
        if k is None:
            k = max(1, p // 3)
        results = []

        for _ in range(n_matrices):
            A, t, b, xi = self._generate_problem(p)
            eta_p, eta_1 = self._get_eigenvalues(A)

            # Algorithm parameters
            epsilon = 0.1 * (k / p)
            delta_0 = (3 * eta_p * epsilon**2) / (1 + 3 * epsilon**2)

            if n_steps > 1:
                r = (eta_1 / delta_0) ** (1 / (n_steps - 1))
            else:
                r = 1

            curr_delta = delta_0
            for _ in range(n_steps):
                results.append(self._calculate_gradient_error(p, curr_delta, t, b, A))
                curr_delta *= r

        return self._process_results("Algo Path", p, results)

    def test_convexity_boundary(self, p, n_matrices):
        """Stress Test: Around eta_1 (Convexity Flip)."""
        results = []
        for _ in range(n_matrices):
            A, t, b, xi = self._generate_problem(p)
            _, eta_1 = self._get_eigenvalues(A)

            for d in [eta_1 * 0.9, eta_1, eta_1 * 1.1]:
                results.append(self._calculate_gradient_error(p, d, t, b, A))

        return self._process_results("Boundary", p, results)

    # --- New Stress Testing Functionality ---

    def run_stress_test(self):
        """
        Systematically tests across different dimensions to verify scalability.
        This simulates the 'Dataset with p=31' vs 'p=85' style testing in your image.
        """
        print("Starting Systematic Stress Test...")

        # Define dimensions to test (Small, Medium, Large)
        dimensions = [5, 20, 50, 100]
        n_matrices = 5  # Matrices per dimension
        n_points = 20  # Points per matrix

        for p in dimensions:
            print(f"Testing Dimension p={p}...")
            # Run all three test suites for this dimension
            self.test_general_random(p, n_matrices, n_points)
            self.test_algorithm_path(p, n_matrices, n_steps=20)
            self.test_convexity_boundary(p, n_matrices)

        # Display final table
        self.logger.display_summary()
        # Optionally save
        # self.logger.save_to_csv("stress_test_results.csv")


if __name__ == "__main__":
    verifier = GradientVerifier()
    verifier.run_stress_test()
