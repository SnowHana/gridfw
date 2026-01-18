import numpy as np
import itertools
from grad_verif.core import ProblemGenerator
from grad_fw.fw_homotomy import FWHomotopySolver
import time
import math


class BruteForceSolver:
    """
    Finds the EXACT global maximum by checking every combination.
    Only use for small p (e.g., p < 25).
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def _calculate_objective(self, indices):
        """
        Calculates Trace( P_s A ) = Trace( (A_ss)^-1 A_s: A_:s )
        """
        # 1. Extract submatrices
        # A_ss: The small k x k square matrix (must be invertible)
        A_ss = self.A[np.ix_(indices, indices)]

        # A_s_all: The k x p rectangular matrix (rows of selected items)
        A_s_all = self.A[indices, :]

        try:
            # 2. Compute Inverse
            A_ss_inv = np.linalg.inv(A_ss)

            # 3. Compute Objective: Trace( A_ss_inv @ A_s_all @ A_s_all.T )
            middle_term = A_s_all @ A_s_all.T
            obj = np.trace(A_ss_inv @ middle_term)
            return obj
        except np.linalg.LinAlgError:
            return -1.0  # Singular matrix, invalid set

    def solve(self):
        print(f"  > Brute Force: Checking {math.comb(self.p, self.k)} combinations...")
        best_obj = -np.inf
        best_indices = None

        # Itertools generates all possible subsets of size k
        for indices in itertools.combinations(range(self.p), self.k):
            indices = list(indices)
            obj = self._calculate_objective(indices)

            if obj > best_obj:
                best_obj = obj
                best_indices = indices

        return np.array(best_indices), best_obj


def test_accuracy_vs_optimal():
    print("=== VALIDATION: Algorithm vs. Brute Force Optimum ===\n")

    # 1. Setup Small Problem (so brute force is fast)
    p = 20  # Keep this small! (Comb(20, 4) = 4,845 checks -> Fast)
    k = 4
    cond = 10.0  # Well-conditioned first to check logic

    gen = ProblemGenerator()
    A = gen.generate_ill_conditioned_matrix(p, cond)

    # 2. Run YOUR Algorithm (FW-Homotopy)
    print(f"1. Running FW-Homotopy (Approximation)...")
    start_time = time.time()
    fw_solver = FWHomotopySolver(A, k, alpha=0.1, n_steps=100, n_mc_samples=50)
    fw_s = fw_solver.solve(verbose=False)
    fw_time = time.time() - start_time

    fw_indices = np.where(fw_s > 0.5)[0]

    # Calculate objective for FW result using the same helper
    bf_checker = BruteForceSolver(A, k)
    fw_obj = bf_checker._calculate_objective(fw_indices)

    print(f"   -> Selected: {np.sort(fw_indices)}")
    print(f"   -> Objective: {fw_obj:.4f}")
    print(f"   -> Time: {fw_time:.4f}s")

    # 3. Run Brute Force (Ground Truth)
    print(f"\n2. Running Brute Force (Exact Optimum)...")
    start_time = time.time()
    opt_indices, opt_obj = bf_checker.solve()
    opt_time = time.time() - start_time

    print(f"   -> Selected: {np.sort(opt_indices)}")
    print(f"   -> Objective: {opt_obj:.4f}")
    print(f"   -> Time: {opt_time:.4f}s")

    # 4. Comparison
    print("\n--- RESULT ---")

    # Calculate Optimality Gap
    # (Opt - FW) / Opt
    gap = (opt_obj - fw_obj) / np.abs(opt_obj) * 100

    if np.array_equal(np.sort(fw_indices), np.sort(opt_indices)):
        print("SUCCESS: Algorithm found the EXACT optimal subset!")
    else:
        print(f"Algorithm found a sub-optimal set.")
        print(f"Optimality Gap: {gap:.4f}%")

        if gap < 1.0:
            print(
                "(This is excellent. <1% gap is usually considered solved for NP-hard problems)"
            )
        else:
            print(
                "WARNING: Gap is significant. Algorithm might be stuck in local optima."
            )


if __name__ == "__main__":
    test_accuracy_vs_optimal()
