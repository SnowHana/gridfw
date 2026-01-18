import pytest
import numpy as np
from grad_verif.verifiers import SingleGradientVerifier


# --- FIXTURE ---
@pytest.fixture
def verifier():
    """Provides a fresh instance of the verifier for each test."""
    return SingleGradientVerifier()


# --- TESTS ---


@pytest.mark.parametrize("p", [50, 100, 200])
@pytest.mark.parametrize("cond", [1e0, 1e4, 1e8])
def test_f_stress_ill_conditioned(verifier, p, cond):
    """test_f_stress_ill_conditioned
    STRESS TEST: Checks gradient accuracy on ill-conditioned matrices.
    Runs 9 separate tests...

    Args:
        verifier (_type_): _description_
        p (_type_): Dimension of matrix
        cond (_type_): Condition number (max to min eigenvalue ratio)
    """
    n_matrices = 2
    n_points = 5

    for _ in range(n_matrices):
        A = verifier.gen.generate_ill_conditioned_matrix(p, cond)
        for _ in range(n_points):
            t, _, b = verifier.gen.generate_vectors(p, A)
            delta = 0.1

            ga = verifier.math.grad_f_analytical(p, delta, t, b, A)
            gn = verifier.math.grad_f_numerical(p, delta, t, b, A, verifier.h)

            err = verifier.check_error(ga, gn)
            assert (
                err < verifier.threshold
            ), f"Gradient mismatch! p={p}, cond={cond:.0e}, err={err:.2e}"


def test_f_algorithm_path(verifier):
    """test_f_algorithm_path
    PATH TEST: Simulates the actual annealing path (delta_0 -> eta_1).
    Ensures gradient stays accurate as delta shrinks.

    Args:
        verifier (_type_): _description_
    """
    p, k, n_steps = 100, 20, 50
    A = verifier.gen.generate_ill_conditioned_matrix(p, 1e2)
    t, _, b = verifier.gen.generate_vectors(p, A)

    # Calculate Schedule
    evals = np.linalg.eigvalsh(A)
    eta_p, eta_1 = max(evals[0], 1e-6), max(evals[-1], 1.0)
    epsilon = 0.1 * (k / p)

    # Robust Delta Start
    delta_0 = max((3 * eta_p * epsilon**2) / (1 + 3 * epsilon**2), 0.1)

    # Geometric Decay
    r = (eta_1 / delta_0) ** (1 / (n_steps - 1)) if delta_0 < eta_1 else 1.0

    curr_delta = delta_0
    for step in range(n_steps):
        ga = verifier.math.grad_f_analytical(p, curr_delta, t, b, A)
        gn = verifier.math.grad_f_numerical(p, curr_delta, t, b, A, verifier.h)
        err = verifier.check_error(ga, gn)

        assert (
            err < verifier.threshold
        ), f"Path failed at Step {step}, delta={curr_delta:.2e}, err={err:.2e}"
        curr_delta *= r


def test_f_convexity_boundary(verifier):
    """
    BOUNDARY TEST: Checks behavior around the critical temperature eta_1.
    """
    p = 50
    A = verifier.gen.generate_ill_conditioned_matrix(p, 1e2)
    t, _, b = verifier.gen.generate_vectors(p, A)

    eta_1 = np.max(np.linalg.eigvalsh(A))

    # Test just below, at, and just above the critical temp
    for delta in [eta_1 * 0.9, eta_1, eta_1 * 1.1]:
        ga = verifier.math.grad_f_analytical(p, delta, t, b, A)
        gn = verifier.math.grad_f_numerical(p, delta, t, b, A, verifier.h)
        err = verifier.check_error(ga, gn)
        assert err < verifier.threshold
