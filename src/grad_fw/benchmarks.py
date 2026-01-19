import numpy as np
import itertools
import math


class GreedySolver:
    """
    Standard Greedy Algorithm for Column Subset Selection.
    Selects one feature at a time to maximize the objective amongst remaining features
    Complexity: O(k * p) steps (approx).
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def calculate_obj(self, indices):
        """
        Calculates the Trace Objective for a given set of indices.
        Obj = Trace( (A_S)^-1 @ (A_S,: @ A_S,:^T) )
        """
        if len(indices) == 0:
            return 0.0

        # Extract Submatrices
        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]  # The k x k small block
        A_s_all = self.A[idx, :]  # The k x p rectangular strip

        try:
            # Note: A_s_all @ A_s_all.T is actually (A^2)_ss
            # But we compute it explicitly here for clarity
            numerator = A_s_all @ A_s_all.T
            return np.trace(np.linalg.inv(A_ss) @ numerator)
        except np.linalg.LinAlgError:
            return -1.0

    def solve(self):
        """
        Runs the Greedy selection process.
        Returns: (best_indices, best_obj, time_taken)
        """
        import time

        start_time = time.time()

        current_indices = []

        # Greedily add k elements
        for _ in range(self.k):
            best_imp_idx = -1
            best_imp_val = -np.inf

            # Try adding every remaining candidate
            candidates = [i for i in range(self.p) if i not in current_indices]

            for candidate in candidates:
                # Test combination
                temp_indices = current_indices + [candidate]
                val = self.calculate_obj(temp_indices)

                if val > best_imp_val:
                    best_imp_val = val
                    best_imp_idx = candidate

            if best_imp_idx != -1:
                current_indices.append(best_imp_idx)

        total_time = time.time() - start_time
        final_obj = self.calculate_obj(current_indices)

        return current_indices, final_obj, total_time


class BruteForceSolver:
    """
    The 'Ground Truth' Solver.
    Checks EVERY possible combination of k features.
    WARNING: Only use for tiny problems (p < 20).
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k
        self.helper = GreedySolver(A, k)  # Reuse calculation logic

    def solve(self, safe_mode=True):
        # 1. Safety Check
        n_combinations = math.comb(self.p, self.k)
        if safe_mode and n_combinations > 2_000_000:
            raise ValueError(
                f"Brute Force is too dangerous! {n_combinations:,} combinations.\n"
                f"Set safe_mode=False to override (Computer might freeze)."
            )

        print(f"Brute Force: Checking {n_combinations} combinations...")

        best_obj = -np.inf
        best_indices = None

        # 2. Iterate ALL combinations
        for indices in itertools.combinations(range(self.p), self.k):
            # Convert tuple to list
            idx_list = list(indices)
            obj = self.helper.calculate_obj(idx_list)

            if obj > best_obj:
                best_obj = obj
                best_indices = idx_list

        return best_indices, best_obj
