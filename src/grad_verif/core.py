import numpy as np


class ProblemGenerator:
    """Generates matrices and vectors for testing."""

    @staticmethod
    def generate_ill_conditioned_matrix(p, condition_number):
        X = np.random.randn(p, p)
        Q, _ = np.linalg.qr(X)
        eigenvalues = np.linspace(1.0, 1.0 / condition_number, p)
        return Q @ np.diag(eigenvalues) @ Q.T

    @staticmethod
    def generate_vectors(p, A, t_bounds=(0.001, 0.99)):
        t = np.random.uniform(t_bounds[0], t_bounds[1], size=p)
        xi = np.random.choice([-1, 1], size=p)
        b = A @ xi
        return t, xi, b


import numpy as np

# ... ProblemGenerator class stays the same ...


class BooleanRelaxation:
    """
    Central repository for all math formulas (f, g, and their gradients).
    Includes numerical stability fixes for t -> 0.
    """

    @staticmethod
    def get_pi_inv(p, delta, t, A):
        """Calculates (A + delta * (T^-2 - I))^-1 with numerical safety."""
        # FIX: Clip t to avoid 1/0 division
        t_safe = np.clip(t, 1e-9, 1.0)

        Dt = np.diag(1.0 / (t_safe**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        return np.linalg.inv(Pi_t)

    # --- Single Function f(t, xi) ---

    @staticmethod
    def grad_f_analytical(p, delta, t, b, A):
        """Formula: 2 * delta * (Pi_inv @ b)^2 / t^3"""
        t_safe = np.clip(t, 1e-9, 1.0)  # FIX

        Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_safe, A)
        return 2 * delta * ((Pi_inv @ b) ** 2) / (t_safe**3)

    @staticmethod
    def grad_f_numerical(p, delta, t, b, A, h=1e-7):
        grad = np.zeros(p)

        def func(t_vec):
            return b.T @ BooleanRelaxation.get_pi_inv(p, delta, t_vec, A) @ b

        for i in range(p):
            t_p, t_m = t.copy(), t.copy()
            t_p[i] += h
            t_m[i] -= h
            grad[i] = (func(t_p) - func(t_m)) / (2 * h)
        return grad

    # --- Expected Function g(t) ---

    @staticmethod
    def grad_g_analytical(p, delta, t, A, xi_samples):
        """
        Computes Mean(grad_f) efficiently.
        """
        t_safe = np.clip(t, 1e-9, 1.0)  # FIX

        Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_safe, A)
        grad_sum = np.zeros(p)

        scale = 2 * delta / (t_safe**3)

        for xi in xi_samples:
            b = A @ xi
            grad_sum += (Pi_inv @ b) ** 2

        return scale * (grad_sum / len(xi_samples))

    @staticmethod
    def grad_g_numerical(p, delta, t, A, xi_samples, h=1e-7):
        grad = np.zeros(p)
        b_vecs = [A @ xi for xi in xi_samples]
        N = len(xi_samples)

        def get_g(t_vec):
            Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_vec, A)
            val = 0.0
            for b in b_vecs:
                val += b.T @ Pi_inv @ b
            return val / N

        for i in range(p):
            t_p, t_m = t.copy(), t.copy()
            t_p[i] += h
            t_m[i] -= h
            grad[i] = (get_g(t_p) - get_g(t_m)) / (2 * h)
        return grad
