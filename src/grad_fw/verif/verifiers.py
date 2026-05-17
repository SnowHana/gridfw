import numpy as np
from grad_fw.verif.core import ProblemGenerator, BooleanRelaxation


class BaseVerifier:
    """
    Helper class for Gradient Verification.
    Provides tools to generate problems and check errors
    """

    def __init__(self, h=1e-7, threshold=1e-5):
        """
        Args:
            h (float): Step size for numerical differentiation.
            threshold (float): Max allowed relative error.
        """
        self.h = h
        self.threshold = threshold
        self.gen = ProblemGenerator()
        self.math = BooleanRelaxation()
        self._results = []  # list of (label, rel_err, passed)

    def check_error(self, g_ana, g_num):
        """
        Calculates relative error between Analytical (g_ana) and Numerical (g_num) gradients.
        Includes safety logic for tiny signals (The 'Relative Error Trap').
        """
        abs_err = np.linalg.norm(g_ana - g_num)
        norm = np.linalg.norm(g_ana)

        # Avoid division by zero
        rel_err = abs_err / (norm + 1e-10)

        # Trust analytical if both signal and error are negligible (e.g., < 1e-8)
        if norm < 1e-8 and abs_err < 1e-8:
            rel_err = 0.0

        return rel_err

    def report(self):
        """Print a summary table of all test results."""
        if not self._results:
            print("No results to report. Run run_stress_test() first.")
            return

        total = len(self._results)
        passed = sum(1 for _, _, ok in self._results if ok)
        print(f"\n{'='*60}")
        print(f"  {self.__class__.__name__} — {passed}/{total} passed")
        print(f"{'='*60}")
        print(f"  {'Case':<30} {'Rel Err':>12} {'Status':>8}")
        print(f"  {'-'*50}")
        for label, rel_err, ok in self._results:
            status = "PASS" if ok else "FAIL"
            print(f"  {label:<30} {rel_err:>12.2e} {status:>8}")
        print(f"{'='*60}\n")


class SingleGradientVerifier(BaseVerifier):
    """Verifies the deterministic gradient grad_f_analytical against grad_f_numerical."""

    TEST_CASES = [
        (5, 1.0, "p=5  cond=1"),
        (10, 10.0, "p=10 cond=10"),
        (15, 100.0, "p=15 cond=100"),
    ]

    def run_stress_test(self):
        """Run gradient checks across several problem sizes and condition numbers."""
        self._results.clear()
        for p, cond, label in self.TEST_CASES:
            np.random.seed(0)
            A = self.gen.generate_ill_conditioned_matrix(p, cond)
            t, xi, b = self.gen.generate_vectors(p, A)
            delta = 0.1

            g_ana = BooleanRelaxation.grad_f_analytical(p, delta, t, b, A)
            g_num = BooleanRelaxation.grad_f_numerical(p, delta, t, b, A, h=self.h)

            rel_err = self.check_error(g_ana, g_num)
            passed = rel_err < self.threshold
            self._results.append((label, rel_err, passed))


class ExpectedGradientVerifier(BaseVerifier):
    """Verifies the stochastic gradient grad_g_analytical against grad_g_numerical."""

    TEST_CASES = [
        (5, 1.0, 20, "p=5  cond=1   m=20"),
        (10, 10.0, 50, "p=10 cond=10  m=50"),
        (15, 100.0, 100, "p=15 cond=100 m=100"),
    ]

    def run_stress_test(self):
        """Run gradient checks across several problem sizes and condition numbers."""
        self._results.clear()
        for p, cond, n_mc, label in self.TEST_CASES:
            np.random.seed(0)
            A = self.gen.generate_ill_conditioned_matrix(p, cond)
            t, _, _ = self.gen.generate_vectors(p, A)
            delta = 0.1
            xi_samples = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]

            g_ana = BooleanRelaxation.grad_g_analytical(p, delta, t, A, xi_samples)
            g_num = BooleanRelaxation.grad_g_numerical(p, delta, t, A, xi_samples, h=self.h)

            rel_err = self.check_error(g_ana, g_num)
            passed = rel_err < self.threshold
            self._results.append((label, rel_err, passed))
