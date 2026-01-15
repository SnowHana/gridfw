import numpy as np
from core import ProblemGenerator, BooleanRelaxation
from logger import GradientTestLogger


class BaseVerifier:
    def __init__(self, h=1e-7, threshold=1e-5, log_prefix="test"):
        self.h = h
        self.threshold = threshold
        self.logger = GradientTestLogger()
        self.log_prefix = log_prefix
        self.gen = ProblemGenerator()
        self.math = BooleanRelaxation()

    def check_error(self, g_ana, g_num):
        """Robust error check handling tiny gradients."""
        abs_err = np.linalg.norm(g_ana - g_num)
        norm = np.linalg.norm(g_ana)

        # Avoid division by zero
        rel_err = abs_err / (norm + 1e-10)

        # Trust analytical if signals are negligible (e.g., < 1e-8)
        if norm < 1e-8 and abs_err < 1e-8:
            rel_err = 0.0

        return rel_err

    def _log_batch(self, name, p, c, errs):
        """Helper to log a batch of errors."""
        mx, mn = np.max(errs), np.mean(errs)
        status = "PASS" if mx < self.threshold else "WARN"
        print(f"[{name:<15}] Dim={p:<4} Cond={c:<7.0e} -> MaxRel: {mx:.2e} ({status})")

        self.logger.log_entry(name, p, c, len(errs), mx, mn, status)

    def report(self):
        """Displays summary and SAVES the log file."""
        self.logger.display_summary()
        self.logger.save_logs(filename_prefix=self.log_prefix)


class SingleGradientVerifier(BaseVerifier):
    def __init__(self):
        super().__init__(log_prefix="f_gradient_verif")

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("Running COMPLETE Verification Suite for f(t)")
        print("=" * 60)
        self.run_stress_test()
        self.run_algorithm_path_test()
        self.run_convexity_boundary_test()

    def run_stress_test(self):
        """Test 1: Ill-Conditioned Matrices (General Robustness)"""
        dims = [50, 100, 200]
        conds = [1e0, 1e4, 1e8]
        n_mats, n_pts = 20, 50  # Rigorous sample size

        print(f"\n[1/3] Stress Test (Ill-Conditioned Matrices)...")
        for p in dims:
            for c in conds:
                errs = []
                for _ in range(n_mats):
                    A = self.gen.generate_ill_conditioned_matrix(p, c)
                    for _ in range(n_pts):
                        t, _, b = self.gen.generate_vectors(p, A)
                        delta = 0.1

                        ga = self.math.grad_f_analytical(p, delta, t, b, A)
                        gn = self.math.grad_f_numerical(p, delta, t, b, A, self.h)
                        errs.append(self.check_error(ga, gn))

                self._log_batch("Stress (Rand)", p, c, errs)

    def run_algorithm_path_test(self):
        """Test 2: Homotopy Path Simulation (delta_0 -> eta_1)"""
        print(f"\n[2/3] Algorithm Path Test (Homotopy Schedule)...")
        p = 100
        n_mats = 20
        n_steps = 50
        k = 20

        errs = []
        for _ in range(n_mats):
            A = self.gen.generate_ill_conditioned_matrix(p, 1e2)
            t, _, b = self.gen.generate_vectors(p, A)

            evals = np.linalg.eigvalsh(A)
            eta_p, eta_1 = evals[0], evals[-1]

            epsilon = 0.1 * (k / p)
            delta_0 = (3 * eta_p * epsilon**2) / (1 + 3 * epsilon**2)
            r = (eta_1 / delta_0) ** (1 / (n_steps - 1))

            curr_delta = delta_0
            for _ in range(n_steps):
                ga = self.math.grad_f_analytical(p, curr_delta, t, b, A)
                gn = self.math.grad_f_numerical(p, curr_delta, t, b, A, self.h)
                errs.append(self.check_error(ga, gn))
                curr_delta *= r

        self._log_batch("Algo Path", p, 1e2, errs)

    def run_convexity_boundary_test(self):
        """Test 3: Convexity Boundary (Around eta_1)"""
        print(f"\n[3/3] Convexity Boundary Test (Around eta_1)...")
        p = 100
        n_mats = 20

        errs = []
        for _ in range(n_mats):
            A = self.gen.generate_ill_conditioned_matrix(p, 1e2)
            t, _, b = self.gen.generate_vectors(p, A)
            evals = np.linalg.eigvalsh(A)
            eta_1 = evals[-1]

            for d in [eta_1 * 0.9, eta_1, eta_1 * 1.1]:
                ga = self.math.grad_f_analytical(p, d, t, b, A)
                gn = self.math.grad_f_numerical(p, d, t, b, A, self.h)
                errs.append(self.check_error(ga, gn))

        self._log_batch("Convexity Bnd", p, 1e2, errs)


class ExpectedGradientVerifier(BaseVerifier):
    def __init__(self):
        super().__init__(log_prefix="g_gradient_verif")

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("Running COMPLETE Verification Suite for g(t) (Expectation)")
        print("=" * 60)
        self.run_stress_test()
        self.run_algorithm_path_test()
        self.run_convexity_boundary_test()

    def run_stress_test(self):
        """Test 1: Ill-Conditioned Matrices for Expectation"""
        # ACADEMIC SCALE: p up to 200, 100 samples per expectation
        dims = [50, 100, 200]
        conds = [1e0, 1e4, 1e8]
        n_mats = 20  # Robust number of matrices
        n_mc = 100  # 100 samples to stress the summation logic

        print(f"\n[1/3] Stress Test g(t) (Ill-Conditioned)...")
        for p in dims:
            for c in conds:
                errs = []
                for _ in range(n_mats):
                    A = self.gen.generate_ill_conditioned_matrix(p, c)
                    xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]
                    t, _, _ = self.gen.generate_vectors(p, A)
                    delta = 0.1

                    ga = self.math.grad_g_analytical(p, delta, t, A, xis)
                    gn = self.math.grad_g_numerical(p, delta, t, A, xis, self.h)
                    errs.append(self.check_error(ga, gn))

                self._log_batch("Stress g(t)", p, c, errs)

    def run_algorithm_path_test(self):
        """Test 2: Homotopy Path for g(t)"""
        print(f"\n[2/3] Algorithm Path Test g(t)...")
        # Scaled up parameters
        p = 100
        n_mats = 20
        n_steps = 50
        k = 20
        n_mc = 50

        errs = []
        for _ in range(n_mats):
            A = self.gen.generate_ill_conditioned_matrix(p, 1e2)
            t, _, _ = self.gen.generate_vectors(p, A)
            xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]

            # Schedule
            evals = np.linalg.eigvalsh(A)
            eta_p, eta_1 = evals[0], evals[-1]
            epsilon = 0.1 * (k / p)
            delta_0 = (3 * eta_p * epsilon**2) / (1 + 3 * epsilon**2)
            r = (eta_1 / delta_0) ** (1 / (n_steps - 1))

            curr_delta = delta_0
            for _ in range(n_steps):
                ga = self.math.grad_g_analytical(p, curr_delta, t, A, xis)
                gn = self.math.grad_g_numerical(p, curr_delta, t, A, xis, self.h)
                errs.append(self.check_error(ga, gn))
                curr_delta *= r

        self._log_batch("Algo Path g(t)", p, 1e2, errs)

    def run_convexity_boundary_test(self):
        """Test 3: Convexity Boundary g(t) around eta_1"""
        print(f"\n[3/3] Convexity Boundary Test g(t)...")
        p = 100
        n_mats = 20
        n_mc = 50

        errs = []
        for _ in range(n_mats):
            A = self.gen.generate_ill_conditioned_matrix(p, 1e2)
            t, _, _ = self.gen.generate_vectors(p, A)
            xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]

            evals = np.linalg.eigvalsh(A)
            eta_1 = evals[-1]

            for d in [eta_1 * 0.9, eta_1, eta_1 * 1.1]:
                ga = self.math.grad_g_analytical(p, d, t, A, xis)
                gn = self.math.grad_g_numerical(p, d, t, A, xis, self.h)
                errs.append(self.check_error(ga, gn))

        self._log_batch("Conv Bnd g(t)", p, 1e2, errs)
