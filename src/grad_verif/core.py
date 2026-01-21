import numpy as np


class ProblemGenerator:
    """Genereate data matrix, vectors for problems

    Returns:
        _type_: _description_
    """

    @staticmethod
    def generate_ill_conditioned_matrix(p, condition_number):
        """generate_ill_conditioned_matrix (Edge cases matrix)

        Args:
            p (int): Dimension of matrix (p x p matrix)
            condition_number (float): Ratio of largest to smallest eigenvaleus (lambda_max / lambda_min)

        Returns:
            npt.NDArray[np.float64]: A (p x p) ill-conditioned matrix
        """
        # Use Eigen-decomposition : A = Q diag(eigen) Q^T
        X = np.random.randn(p, p)
        # QR Decomposition: A = QR (Q is an orthogonal matrix)
        Q, _ = np.linalg.qr(X)
        eigenvalues = np.linspace(1.0, 1.0 / condition_number, p)
        return Q @ np.diag(eigenvalues) @ Q.T

    @staticmethod
    def generate_vectors(p, A, t_bounds=(0.001, 0.999)):
        """generate_vectors Generate random problem vectors t, xi, and b

        Args:
            p (int): Dimension of A (p x p)
            A (_type_): p x p data matrix
            t_bounds (tuple[float, float]): Lower and Upper bound of t. Defaults to (0.001, 0.999).

        Returns:
            _type_: _description_
        """
        t = np.random.uniform(t_bounds[0], t_bounds[1], size=p)
        # Rademcaher vector of dimension p : [-1, 1]
        xi = np.random.choice([-1, 1], size=p)
        b = A @ xi
        return t, xi, b


class BooleanRelaxation:
    """Central Repo
    Boolean Relaxation from Scalable-Gradient

    Returns:
        _type_: _description_
    """

    @staticmethod
    def get_pi_inv(p, delta, t, A):
        """
        Docstring for get_pi_inv
        Calculates (A + delta * (T^-2 - I))^-1 with numerical safety.

        :param p: Dimension of data matrix
        :param delta: "Tuning" Parameter
        :param t: Current point in (0, 1)
        :param A: Gram matrix of Data matrix (A = X^T @ X)
        """

        # NOTE: Clip t to make sure it stays in our range
        # cuz t can "marginally" exceed the range
        t_safe = np.clip(t, 1e-9, 1.0)

        # Dt = T^-2 - I
        Dt = np.diag(1.0 / (t_safe**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        return np.linalg.inv(Pi_t)

    # --- Single Function f(t, xi) ---

    @staticmethod
    def grad_f_analytical(p, delta, t, b, A):
        """
        Docstring for grad_f_analytical
        Return grad_f based on a FORMULA
        Formula: 2 * delta * (Pi_inv @ b)^2 / t^3

        :param p: Description
        :param delta: Description
        :param t: Interior point
        :param b: Rademacher vector controlled
        :param A: Description
        """
        t_safe = np.clip(t, 1e-9, 1.0)  # FIX
        Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_safe, A)
        return 2 * delta * ((Pi_inv @ b) ** 2) / (t_safe**3)

    @staticmethod
    def grad_f_numerical(p, delta, t, b, A, h=1e-7):
        """grad_f_numerical : Calcualte gradient NUMERICALLY
        Central Difference Method
        Formula: df(x) / dx

        Args:
            p (_type_): _description_
            delta (_type_): _description_
            t (_type_): _description_
            b (_type_): _description_
            A (_type_): _description_
            h (_type_, optional): _description_. Defaults to 1e-7.

        Returns:
            _type_: _description_
        """
        grad = np.zeros(p)

        def func(t_vec):
            # f = b.T @ Pi^inv @ b
            return b.T @ BooleanRelaxation.get_pi_inv(p, delta, t_vec, A) @ b

        for i in range(p):
            # Numerical change of a single component
            t_p, t_m = t.copy(), t.copy()
            t_p[i] += h
            t_m[i] -= h
            grad[i] = (func(t_p) - func(t_m)) / (2 * h)
        return grad

    # --- Expected Function g(t) ---

    @staticmethod
    def grad_g_analytical(p, delta, t, A, xi_samples):
        """grad_g_analytical Computes gradient of g
        (Estimated expectation of f over subset of Rademacher vectors)
        """
        t_safe = np.clip(t, 1e-9, 1.0)
        
        # Dt = T^-2 - I
        Dt = np.diag(1.0 / (t_safe**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        
        # Batch solve: Pi_t @ X = B, where B = A @ xi_samples.T
        # This is O(p^3 + p^2 * N) instead of O(p^3 + p^2 * N) but with better constants
        # and avoids explicit inversion.
        B = A @ np.array(xi_samples).T
        X = np.linalg.solve(Pi_t, B)
        
        # Gradient component j is (2 * delta / t_j^3) * mean(X_j^2)
        grad_sum = np.sum(X**2, axis=1)
        scale = 2 * delta / (t_safe**3)
        
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

    # --- Convenience Methods for Minimization (h = -f, z = -g) ---

    @staticmethod
    def grad_h_analytical(p, delta, t, b, A):
        """
        Gradient of h = -f.
        Used for MINIMIZATION problems.
        """
        return -BooleanRelaxation.grad_f_analytical(p, delta, t, b, A)

    @staticmethod
    def grad_z_analytical(p, delta, t, A, xi_samples):
        """
        Gradient of z = -g.
        Used for MINIMIZATION problems.
        """
        return -BooleanRelaxation.grad_g_analytical(p, delta, t, A, xi_samples)

    @staticmethod
    def grad_a_opt_analytical(p, delta, t, A, xi_samples):
        """
        Gradient for A-Optimality (Trace of Inverse).
        Objective: E[ xi^T (A + delta*Dt)^-1 xi ]
        This is what we want to MINIMIZE.
        """
        t_safe = np.clip(t, 1e-9, 1.0)
        
        Dt = np.diag(1.0 / (t_safe**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        
        # Batch solve: Pi_t @ X = xi_samples.T
        B = np.array(xi_samples).T
        X = np.linalg.solve(Pi_t, B)
        
        grad_sum = np.sum(X**2, axis=1)
        scale = 2 * delta / (t_safe**3)

        return scale * (grad_sum / len(xi_samples))

    @staticmethod
    def grad_portfolio_analytical(p, delta, t, A):
        """
        Gradient for Minimum-Variance Portfolio (Maximizing 1^T A_S^-1 1).
        Objective: 1^T (A + delta*Dt)^-1 1
        This is a MAXIMIZATION problem.
        Returns NEGATIVE gradient for minimization solver.
        """
        b = np.ones(p)
        # Gradient of f(t) is positive (increasing t increases objective)
        grad_f = BooleanRelaxation.grad_f_analytical(p, delta, t, b, A)
        return -grad_f
