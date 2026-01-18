import numpy as np
import time


class GreedySolver:
    """
    The Industry Standard Benchmark for CSSP.
    Fast, reliable, and what reviewers expect you to beat.
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def solve(self):
        """Returns: indices, objective_value, time_taken"""
        selected = []
        available = list(range(self.p))

        start_time = time.time()

        # Iteratively pick the single best column to add
        for step in range(self.k):
            best_idx = -1
            best_gain = -np.inf

            for idx in available:
                # Test "What if we add idx?"
                current_set = selected + [idx]

                # --- FAST OBJECTIVE CALCULATION ---
                # A_ss = Submatrix of chosen items
                A_ss = self.A[np.ix_(current_set, current_set)]
                A_s_all = self.A[current_set, :]

                try:
                    # Obj = Trace( (A_ss)^-1 @ A_s_all @ A_s_all.T )
                    # We use pinv for stability during the greedy build-up
                    val = np.trace(np.linalg.pinv(A_ss) @ (A_s_all @ A_s_all.T))
                except np.linalg.LinAlgError:
                    val = -1.0

                if val > best_gain:
                    best_gain = val
                    best_idx = idx

            if best_idx != -1:
                selected.append(best_idx)
                available.remove(best_idx)
            else:
                break

        total_time = time.time() - start_time
        return np.array(selected), best_gain, total_time

    # Helper to verify your FW result against Greedy's logic
    def calculate_obj(self, indices):
        indices = list(indices)
        if len(indices) == 0:
            return 0.0
        A_ss = self.A[np.ix_(indices, indices)]
        A_s_all = self.A[indices, :]
        return np.trace(np.linalg.pinv(A_ss) @ (A_s_all @ A_s_all.T))
