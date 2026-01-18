import pytest
import numpy as np
from grad_verif.verifiers import ExpectedGradientVerifier


# --- FIXTURE ---
@pytest.fixture
def verifier():
    return ExpectedGradientVerifier()


# --- STANDARD STRESS TESTS ---


@pytest.mark.parametrize("p", [50, 100, 200])
@pytest.mark.parametrize("cond", [1e2, 1e5, 1e8])
def test_g_stress_ill_conditioned(verifier, p, cond):
    """
    STANDARD STRESS: Checks Expected Gradient across standard ranges.
    Includes high condition numbers (1e8) to test numerical stability.
    """
    n_matrices = 2
    n_mc_samples = 30

    for _ in range(n_matrices):
        A = verifier.gen.generate_ill_conditioned_matrix(p, cond)
        xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc_samples)]

        t, _, _ = verifier.gen.generate_vectors(p, A)
        delta = 0.1

        ga = verifier.math.grad_g_analytical(p, delta, t, A, xis)
        gn = verifier.math.grad_g_numerical(p, delta, t, A, xis, verifier.h)

        err = verifier.check_error(ga, gn)
        assert (
            err < verifier.threshold
        ), f"Standard Stress Fail: p={p}, cond={cond:.0e}, err={err:.2e}"


# --- EDGE CASE TESTS ---
def test_g_boundary_values(verifier):
    """
    EDGE CASE: What happens when t is very close to 0 or 1?
    Fixed: Used 1e-6 instead of 1e-10 to respect step size h=1e-7.
    """
    p = 50
    A = verifier.gen.generate_ill_conditioned_matrix(p, 10.0)
    xis = [np.random.choice([-1, 1], size=p) for _ in range(10)]
    delta = 0.1

    # Case 1: t close to 0 (Near empty set)
    # MUST be larger than verifier.h (1e-7) to avoid clipping artifacts
    t_zero = np.full(p, 1e-6)
    ga = verifier.math.grad_g_analytical(p, delta, t_zero, A, xis)
    gn = verifier.math.grad_g_numerical(p, delta, t_zero, A, xis, verifier.h)
    assert verifier.check_error(ga, gn) < verifier.threshold, "Failed near t=0"

    # Case 2: t close to 1 (Near full set)
    t_one = np.full(p, 1.0 - 1e-6)
    ga = verifier.math.grad_g_analytical(p, delta, t_one, A, xis)
    gn = verifier.math.grad_g_numerical(p, delta, t_one, A, xis, verifier.h)
    assert verifier.check_error(ga, gn) < verifier.threshold, "Failed near t=1"

    # Case 3: Mixed (Some 0, Some 1)
    t_mixed = np.array([1e-6 if i % 2 == 0 else 1.0 - 1e-6 for i in range(p)])
    ga = verifier.math.grad_g_analytical(p, delta, t_mixed, A, xis)
    gn = verifier.math.grad_g_numerical(p, delta, t_mixed, A, xis, verifier.h)
    assert (
        verifier.check_error(ga, gn) < verifier.threshold
    ), "Failed at mixed boundaries"


def test_g_extreme_conditioning(verifier):
    """
    EXTREME CASE: Matrix is almost singular (Condition Number 1e12).
    This ensures we don't crash when A is barely invertible.
    """
    p = 30
    cond = 1e12  # Brutal condition number
    A = verifier.gen.generate_ill_conditioned_matrix(p, cond)
    xis = [np.random.choice([-1, 1], size=p) for _ in range(20)]
    t, _, _ = verifier.gen.generate_vectors(p, A)
    delta = 0.05  # Small delta makes it harder

    ga = verifier.math.grad_g_analytical(p, delta, t, A, xis)
    gn = verifier.math.grad_g_numerical(p, delta, t, A, xis, verifier.h)

    # We allow a slightly looser threshold for such extreme cases
    err = verifier.check_error(ga, gn)
    assert err < 1e-4, f"Extreme Cond Failed: err={err:.2e}"


# --- SCALABILITY TEST ---


@pytest.mark.slow
def test_g_high_dimensions(verifier):
    """
    SCALE TEST: Larger dimensions (p=400).
    Note: Numerical gradient is O(p^4) or worse, so this will be slow.
    Marked as 'slow' so you can skip it if needed.
    """
    p = 400
    cond = 1e2
    n_mc_samples = 10

    A = verifier.gen.generate_ill_conditioned_matrix(p, cond)
    xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc_samples)]
    t, _, _ = verifier.gen.generate_vectors(p, A)
    delta = 0.1

    ga = verifier.math.grad_g_analytical(p, delta, t, A, xis)
    # Only compute numerical for first 5 dims to save time, but check shape
    gn_partial = np.zeros(p)

    # Full verification is too slow, so we do a "Spot Check"
    # Verify just indices 0, 10, 100, 399
    indices_to_check = [0, 10, 100, p - 1]

    for idx in indices_to_check:
        # Manually compute numerical grad for just this index
        t_p, t_m = t.copy(), t.copy()
        t_p[idx] += verifier.h
        t_m[idx] -= verifier.h

        # Helper to calc objective
        def get_obj(vec):
            val = 0
            Pi_inv = verifier.math.get_pi_inv(p, delta, vec, A)
            for xi in xis:
                b = A @ xi
                val += b.T @ Pi_inv @ b
            return val / n_mc_samples

        num_val = (get_obj(t_p) - get_obj(t_m)) / (2 * verifier.h)
        ana_val = ga[idx]

        diff = abs(num_val - ana_val)
        norm = abs(ana_val) + 1e-10
        rel = diff / norm

        assert rel < verifier.threshold, f"High Dim Spot Check Failed at idx={idx}"


# --- ALGORITHM SIMULATION ---


def test_g_algorithm_path(verifier):
    """
    PATH TEST: Full simulation of the annealing process.
    """
    p, k, n_steps = 50, 10, 20
    n_mc = 30

    A = verifier.gen.generate_ill_conditioned_matrix(p, 1e2)
    t, _, _ = verifier.gen.generate_vectors(p, A)
    xis = [np.random.choice([-1, 1], size=p) for _ in range(n_mc)]

    evals = np.linalg.eigvalsh(A)
    eta_1 = max(evals[-1], 1.0)

    delta_0 = 0.1
    r = (eta_1 / delta_0) ** (1 / (n_steps - 1)) if delta_0 < eta_1 else 1.0

    curr_delta = delta_0
    for step in range(n_steps):
        ga = verifier.math.grad_g_analytical(p, curr_delta, t, A, xis)
        gn = verifier.math.grad_g_numerical(p, curr_delta, t, A, xis, verifier.h)
        err = verifier.check_error(ga, gn)

        assert err < verifier.threshold, f"Path Fail at Step {step}"
        curr_delta *= r
