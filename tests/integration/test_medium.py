import pytest
import numpy as np
from grad_fw.benchmarks import GreedySolver
from grad_fw.fw_homotomy import FWHomotopySolver


def test_medium_stability():
    """
    Medium-sized test to verify numerical stability.
    p=100, k=20.
    Checks if Greedy solver produces reasonable objective values (not 1e17).
    """
    p = 100
    k = 20
    np.random.seed(42)

    # Generate synthetic data
    # Make it slightly ill-conditioned to stress test the solver
    X = np.random.randn(200, p)
    # Introduce some correlation
    X[:, 1] = X[:, 0] + 1e-5 * np.random.randn(200)
    A = X.T @ X

    print(f"\n--- Medium Test (p={p}, k={k}) ---")

    # 1. Greedy
    greedy = GreedySolver(A, k)
    idx_g, obj_g, time_g = greedy.solve()
    print(f"Greedy: Obj={obj_g:.4f}, Time={time_g:.4f}s")

    # Check for explosion
    assert obj_g < 1e10, f"Greedy objective exploded: {obj_g}"
    assert obj_g > 0, "Greedy objective should be positive"

    # 2. FW-Homotopy
    fw = FWHomotopySolver(A, k, n_steps=200, n_mc_samples=20)
    s_fw = fw.solve(verbose=False)

    idx_fw = np.where(s_fw > 0.5)[0]
    obj_fw = greedy.calculate_obj(list(idx_fw))
    print(f"FW: Obj={obj_fw:.4f}")

    assert obj_fw < 1e10, f"FW objective exploded: {obj_fw}"

    print("Test Passed: Values are stable.")


if __name__ == "__main__":
    test_medium_stability()
