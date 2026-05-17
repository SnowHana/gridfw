import numpy as np
from scipy.optimize import check_grad
from grad_fw.verif.core import BooleanRelaxation, ProblemGenerator


def test_scipy_check_f():
    """
    Verify grad_f using scipy.optimize.check_grad.
    This provides an independent check of our analytical formula against
    Scipy's finite difference approximation.
    """
    p = 50
    delta = 0.1
    cond = 100.0

    # Generate problem
    A = ProblemGenerator.generate_ill_conditioned_matrix(p, cond)
    t, _, b = ProblemGenerator.generate_vectors(p, A)

    # Define function f(t)
    def func(t_vec):
        # f(t) = b^T (A + delta * (T^-2 - I))^-1 b
        Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_vec, A)
        return b.T @ Pi_inv @ b

    # Define gradient grad_f(t)
    def grad(t_vec):
        return BooleanRelaxation.grad_f_analytical(p, delta, t_vec, b, A)

    # Scipy check_grad returns the norm of the difference
    # It uses forward difference by default with epsilon=1.49e-08
    err = check_grad(func, grad, t)

    print(f"Scipy check_grad error for f(t): {err:.2e}")

    # Threshold: Scipy's default epsilon is small, so error should be small (~1e-6 or less)
    assert err < 1e-5, f"Scipy check failed for f(t)! Error: {err:.2e}"


def test_scipy_check_g():
    """
    Verify grad_g using scipy.optimize.check_grad.
    """
    p = 30  # Smaller p for speed
    delta = 0.1
    cond = 100.0
    n_mc = 10

    A = ProblemGenerator.generate_ill_conditioned_matrix(p, cond)
    t, _, _ = ProblemGenerator.generate_vectors(p, A)
    xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]
    b_vecs = [A @ xi for xi in xis]

    # Define function g(t) = E[f(t)]
    def func(t_vec):
        Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t_vec, A)
        val = 0.0
        for b in b_vecs:
            val += b.T @ Pi_inv @ b
        return val / n_mc

    # Define gradient grad_g(t)
    def grad(t_vec):
        return BooleanRelaxation.grad_g_analytical(p, delta, t_vec, A, xis)

    err = check_grad(func, grad, t)

    print(f"Scipy check_grad error for g(t): {err:.2e}")

    assert err < 1e-5, f"Scipy check failed for g(t)! Error: {err:.2e}"


if __name__ == "__main__":
    # Allow running directly
    try:
        test_scipy_check_f()
        test_scipy_check_g()
        print("All Scipy checks passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
