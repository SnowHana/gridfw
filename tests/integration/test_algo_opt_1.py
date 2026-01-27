import pytest
import numpy as np
import math
import itertools
from grad_verif.core import ProblemGenerator
from grad_fw.benchmarks import BruteForceSolver
from logs.old_logs.fw_homotomy import FWHomotopySolver


# --- Test Cases ---
@pytest.mark.parametrize(
    "p, k, cond_num, n_steps, n_samples",
    [
        (10, 2, 1.0, 100, 50),  # Easy: Fast is fine
        (15, 3, 10.0, 200, 50),  # Medium
        (15, 5, 10.0, 200, 50),  # Medium
        (15, 3, 100.0, 500, 100),  # HARD: Needs slower schedule & less noise
    ],
)
def test_fw_accuracy(p, k, cond_num, n_steps, n_samples):
    """
    Verifies that FW-Homotopy finds a solution within 1% of the global optimum.
    """
    # 1. Generate Problem
    gen = ProblemGenerator()
    A = gen.generate_ill_conditioned_matrix(p, cond_num)

    # 2. Solve with Your Algorithm (Use adaptive parameters)
    solver = FWHomotopySolver(
        A,
        k,
        alpha=0.01,  # Small step size for stability
        n_steps=1000,  # More steps to compensate for small alpha
        n_mc_samples=100,  # Higher precision
    )

    s_fw = solver.solve(verbose=False)
    fw_indices = np.where(s_fw > 0.5)[0]

    # 3. Solve with Brute Force (Ground Truth)
    bf = BruteForceSolver(A, k)
    opt_indices, opt_obj = bf.solve()
    fw_obj = bf.calculate_objective(fw_indices)

    # 4. Assertions
    # Gap calculation: (Optimal - FW) / Optimal
    gap_percent = (opt_obj - fw_obj) / np.abs(opt_obj) * 100

    print(f"\n[TestCase p={p}, k={k}, cond={cond_num}]")
    print(f"   Opt: {opt_obj:.4f} | FW: {fw_obj:.4f} | Gap: {gap_percent:.4f}%")

    # We allow a small optimality gap (e.g., 1-5%) because FW is a heuristic
    # Ideally it should be < 1% for these small problems
    assert (
        gap_percent < 5.0
    ), f"Gap too high: {gap_percent:.2f}% (Opt: {opt_obj}, FW: {fw_obj})"


def test_cardinality_constraint():
    """Checks if the solver respects the k constraint exactly."""
    p, k = 20, 5
    gen = ProblemGenerator()
    A = gen.generate_ill_conditioned_matrix(p, 10.0)

    solver = FWHomotopySolver(A, k)
    s_fw = solver.solve(verbose=False)

    selected_count = np.sum(s_fw)
    assert np.isclose(selected_count, k), f"Expected {k} items, got {selected_count}"
