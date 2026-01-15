import numpy as np
import pandas as pd
import time
import os


class GradientTestLogger:
    """Handles logging of test results, summary generation, and file saving."""

    def __init__(self, output_dir="logs"):
        self.logs = []
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def log_entry(
        self, test_name, p, cond_num, n_samples, max_rel_err, mean_rel_err, status
    ):
        """Records a single test configuration result."""
        self.logs.append(
            {
                "Test": test_name,
                "Dim": p,
                "Cond(A)": f"{cond_num:.1e}",
                "Samples": n_samples,
                "Max Rel": f"{max_rel_err:.2e}",
                "Mean Rel": f"{mean_rel_err:.2e}",
                "Status": status,
            }
        )

    def display_summary(self):
        """Prints a pandas DataFrame summary to the console."""
        if not self.logs:
            print("No logs to display.")
            return

        df = pd.DataFrame(self.logs)
        print("\n" + "=" * 90)
        print(f"STRESS TEST SUMMARY REPORT")
        print("=" * 90)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))
        print("=" * 90 + "\n")

    def save_logs(self, filename_prefix="f_gradient_stress_test"):
        """Saves the current logs to a CSV file with a timestamp."""
        if not self.logs:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        df = pd.DataFrame(self.logs)
        df.to_csv(filepath, index=False)
        print(f"Detailed logs saved to: {filepath}")


class ProblemGenerator:
    """Encapsulates all logic for generating matrices and vectors."""

    @staticmethod
    def generate_ill_conditioned_matrix(p, condition_number):
        """Generates a PD matrix A = X^T X with a specific condition number."""
        # 1. Random orthogonal matrix Q via QR decomposition
        X = np.random.randn(p, p)
        Q, _ = np.linalg.qr(X)

        # 2. Force eigenvalues to spread from 1.0 down to 1/cond
        eigenvalues = np.linspace(1.0, 1.0 / condition_number, p)
        S = np.diag(eigenvalues)

        # 3. Reconstruct A = Q * S * Q^T
        A = Q @ S @ Q.T
        return A

    @staticmethod
    def generate_vectors(p, A, t_bounds=(0.001, 0.99)):
        """Generates random interior point t, Rademacher vector xi, and vector b."""
        # Generate t slightly away from 0/1 to avoid singularity, but close enough to stress test
        t = np.random.uniform(t_bounds[0], t_bounds[1], size=p)

        # [cite_start]Rademacher vector xi in {-1, 1} [cite: 323, 324]
        xi = np.random.choice([-1, 1], size=p)

        # [cite_start]b^(xi) := A * xi (since A = X^T X) [cite: 315]
        b = A @ xi

        return t, xi, b


class GradientVerifier:
    """Orchestrates the verification process using Generator and Logger."""

    def __init__(self, h=1e-7, rel_threshold=1e-5):
        self.h = h
        self.threshold = rel_threshold
        self.logger = GradientTestLogger()
        self.generator = ProblemGenerator()

    def _compute_analytical_gradient(self, p, delta, t, b, A):
        """Calculates gradient using the derived formula[cite: 407, 484]."""
        # Pi_t = A + delta * (T_t^-2 - I)
        Dt = np.diag(1.0 / (t**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        Pi_inv = np.linalg.inv(Pi_t)

        # Formula: 2 * delta * (Pi_inv @ b)^2 / t^3
        # Note: Operations are element-wise
        return 2 * delta * ((Pi_inv @ b) ** 2) / (t**3)

    def _compute_numerical_gradient(self, p, delta, t, b, A):
        """Calculates gradient using central finite difference."""
        grad_numerical = np.zeros(p)

        def get_f_val(t_vec):
            # Must recompute Dt and Inverse for every perturbation
            Dt = np.diag(1.0 / (t_vec**2)) - np.eye(p)
            Pi_t = A + delta * Dt
            return b.T @ np.linalg.inv(Pi_t) @ b

        for i in range(p):
            t_plus = t.copy()
            t_plus[i] += self.h
            t_minus = t.copy()
            t_minus[i] -= self.h
            grad_numerical[i] = (get_f_val(t_plus) - get_f_val(t_minus)) / (2 * self.h)

        return grad_numerical

    def _compare_gradients(self, grad_analytical, grad_numerical):
        """Computes robust error metrics between the two gradients."""
        abs_error = np.linalg.norm(grad_analytical - grad_numerical)
        grad_norm = np.linalg.norm(grad_analytical)

        # Standard relative error
        rel_error = abs_error / (grad_norm + 1e-10)

        # Robustness check: Trust analytical if gradient is negligible (e.g., < 1e-8)
        # This handles the "small delta" case in Algorithm Path simulations
        if grad_norm < 1e-8 and abs_error < 1e-8:
            rel_error = 0.0

        return rel_error

    def run_single_configuration(self, p, cond_num, n_matrices, n_points):
        """Runs a batch of tests for a specific dimension and condition number."""
        all_rel_errors = []

        for _ in range(n_matrices):
            # 1. Generate Matrix A
            A = self.generator.generate_ill_conditioned_matrix(p, cond_num)

            for _ in range(n_points):
                # 2. Generate vectors t, xi, b
                t, xi, b = self.generator.generate_vectors(p, A)

                # 3. Set delta (using a smallish value to stress the formula)
                delta = 0.1

                # 4. Compute Gradients
                grad_ana = self._compute_analytical_gradient(p, delta, t, b, A)
                grad_num = self._compute_numerical_gradient(p, delta, t, b, A)

                # 5. Compare
                error = self._compare_gradients(grad_ana, grad_num)
                all_rel_errors.append(error)

        return np.array(all_rel_errors)

    def run_stress_test_suite(self):
        """Main loop that iterates through dimensions and condition numbers."""
        # Configuration
        dimensions = [50, 100, 200]
        condition_numbers = [1e0, 1e4, 1e8]
        n_matrices = 20
        n_points = 50

        total_tests = len(dimensions) * len(condition_numbers) * n_matrices * n_points
        print(f"Starting Stress Test Suite: {total_tests} total gradients...")

        for p in dimensions:
            for cond in condition_numbers:
                # Run the batch
                errors = self.run_single_configuration(p, cond, n_matrices, n_points)

                # Aggregate stats
                max_err = np.max(errors)
                mean_err = np.mean(errors)
                status = "PASS" if max_err < self.threshold else "WARN"

                print(f"Completed p={p}, Cond={cond:.0e} -> Max Rel: {max_err:.2e}")

                # Log
                self.logger.log_entry(
                    test_name="Stress",
                    p=p,
                    cond_num=cond,
                    n_samples=len(errors),
                    max_rel_err=max_err,
                    mean_rel_err=mean_err,
                    status=status,
                )

        # Finalize
        self.logger.display_summary()
        self.logger.save_logs()


if __name__ == "__main__":
    verifier = GradientVerifier()
    verifier.run_stress_test_suite()
