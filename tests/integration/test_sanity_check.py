import pytest
import numpy as np
import itertools
from grad_verif.core import ProblemGenerator
from logs.old_logs.fw_homotomy import FWHomotopySolver


# --- HELPER: Local Brute Force Solver (Ground Truth) ---
class SimpleBruteForce:
    """
    Exact solver for small problems.
    Tries ALL combinations to find the global minimum of Trace(Inv(A_S)).
    """

    def __init__(self, A, k):
        self.A = A
        self.k = k
        self.p = A.shape[0]

    def solve(self):
        # Initialize with Infinity (since we are MINIMIZING)
        best_obj = np.inf
        best_idx = None

        # Try all combinations of size k
        for indices in itertools.combinations(range(self.p), self.k):
            idx = list(indices)
            A_sub = self.A[np.ix_(idx, idx)]
            try:
                # We want to MINIMIZE the Trace of the Pseudo-Inverse
                obj = np.trace(np.linalg.pinv(A_sub))
            except np.linalg.LinAlgError:
                obj = np.inf

            # Check if we found a better (smaller) objective
            if obj < best_obj:
                best_obj = obj
                best_idx = idx

        return np.array(best_idx), best_obj

    def calculate_obj(self, indices):
        """Helper to verify FW's objective on the same scale."""
        idx = list(indices)
        if len(idx) == 0:
            return np.inf
        A_sub = self.A[np.ix_(idx, idx)]
        try:
            return np.trace(np.linalg.pinv(A_sub))
        except np.linalg.LinAlgError:
            return np.inf


# --- TEST 1: The "Obvious Answer" (Diagonal Sanity Check) ---
def test_diagonal_sanity():
    """
    Verifies the gradient logic.
    For A = diag(100, 100, 1, 1...), minimizing Trace(Inv(A_S)) means
    picking the LARGEST eigenvalues (100, 100), because 1/100 < 1/1.
    """
    print("\n[Sanity] Running Diagonal Matrix Test...")

    # Create diagonal values: Two huge ones, rest tiny.
    # Indices 0 and 1 are the "correct" answer.
    diag_values = np.array([100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    A = np.diag(diag_values)
    k = 2

    # Solve with FW
    # Note: No restarts needed for convex diagonal problem
    solver = FWHomotopySolver(A, k, n_steps=200, n_mc_samples=50)
    s_fw = solver.solve(n_restarts=1, verbose=False)

    # Extract indices
    indices = np.where(s_fw > 0.5)[0]
    indices.sort()

    print(f"  > Selected Indices: {indices}")
    print(f"  > Expected Indices: [0 1]")

    # Hard Assertion: It must be exactly [0, 1]
    assert np.array_equal(
        indices, np.array([0, 1])
    ), f"Failed Obvious Test! Expected [0, 1], got {indices}"


# --- TEST 2: Accuracy Benchmark (Random Matrices) ---
@pytest.mark.parametrize(
    "p, k, cond_num, n_steps, n_samples",
    [
        (10, 2, 1.0, 200, 20),  # Easy: Well-conditioned
        (12, 3, 10.0, 500, 50),  # Medium: Ill-conditioned
        (12, 4, 50.0, 1000, 50),  # Hard: Very Ill-conditioned
    ],
)
def test_fw_vs_bruteforce(p, k, cond_num, n_steps, n_samples):
    """
    Verifies FW finds the global optimum (or very close) on small random matrices.
    Uses Random Restarts for harder cases to avoid local minima.
    """
    # 1. Generate Problem
    # Fix seed for reproducibility!
    np.random.seed(42)
    gen = ProblemGenerator()
    A = gen.generate_ill_conditioned_matrix(p, cond_num)

    # 2. Solve with FW
    # Use restarts for harder problems (cond_num > 1)
    restarts = 5 if cond_num > 1.0 else 1

    solver = FWHomotopySolver(A, k, n_steps=n_steps, n_mc_samples=n_samples)
    s_fw = solver.solve(n_restarts=restarts, verbose=False)
    fw_indices = np.where(s_fw > 0.5)[0]

    # 3. Solve with Brute Force (Ground Truth)
    bf = SimpleBruteForce(A, k)
    opt_indices, opt_obj = bf.solve()

    # Recalculate FW objective using the exact same method as BruteForce
    fw_obj = bf.calculate_obj(fw_indices)

    # 4. Calculate Gap for MINIMIZATION
    # Ideally FW_Obj >= Opt_Obj (since Opt is the global minimum)
    # Gap = (FW - Opt) / |Opt| * 100
    gap = (fw_obj - opt_obj) / abs(opt_obj) * 100

    print(f"\n[Test p={p}, k={k}, cond={cond_num}]")
    print(f"  > Opt Obj: {opt_obj:.4f}")
    print(f"  > FW Obj:  {fw_obj:.4f}")
    print(f"  > Gap:     {gap:.2f}%")

    # Assertion: Allow 5% gap (FW is heuristic vs Exact)
    # Random matrices are non-convex, so small deviations are expected.
    assert gap < 5.0, f"Gap too high: {gap:.2f}%"


# --- TEST 3: Constraint Satisfaction ---
def test_cardinality_constraint():
    """Checks if the solver respects the k constraint exactly."""
    p, k = 20, 5
    gen = ProblemGenerator()
    A = gen.generate_ill_conditioned_matrix(p, 10.0)

    solver = FWHomotopySolver(A, k, n_steps=200)
    s_fw = solver.solve(verbose=False)

    selected_count = np.sum(s_fw > 0.5)
    assert np.isclose(selected_count, k), f"Expected {k} items, got {selected_count}"
