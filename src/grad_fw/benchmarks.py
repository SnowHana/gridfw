import numpy as np
import itertools
import time


class GreedySolver:
    """
    Standard Greedy Algorithm for A-Optimality (Minimizing Trace of Inverse).
    Selects one feature at a time to MINIMIZE the objective.
    Complexity: O(k * p) steps.
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def calculate_obj(self, indices):
        """
        Calculates the A-Optimality Objective: Trace( (A_S)^-1 )
        We use pseudo-inverse (pinv) for stability.
        """
        if len(indices) == 0:
            return np.inf

        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]  # The selected submatrix

        try:
            # Objective: MINIMIZE the Trace of the Inverse
            return np.trace(np.linalg.pinv(A_ss))
        except np.linalg.LinAlgError:
            return np.inf

    def solve(self):
        """
        Runs the Greedy selection process.
        Returns: (best_indices, best_obj, time_taken)
        """
        start_time = time.time()
        current_indices = []

        # Greedily add k elements
        for _ in range(self.k):
            best_idx = -1
            best_obj = np.inf

            # Try adding every remaining candidate
            candidates = [i for i in range(self.p) if i not in current_indices]

            for candidate in candidates:
                # Test combination
                temp_indices = current_indices + [candidate]

                # Calculate objective (We want the MINIMUM value)
                val = self.calculate_obj(temp_indices)

                if val < best_obj:
                    best_obj = val
                    best_idx = candidate

            if best_idx != -1:
                current_indices.append(best_idx)

        total_time = time.time() - start_time
        final_obj = self.calculate_obj(current_indices)

        return np.array(current_indices), final_obj, total_time


class BruteForceSolver:
    """
    Exact solver that tries ALL combinations to find the global minimum.
    Complexity: O(p choose k) - Only for small p!
    """

    def __init__(self, A, k):
        self.A = A
        self.k = k
        self.p = A.shape[0]

    def calculate_obj(self, indices):
        idx = list(indices)
        A_sub = self.A[np.ix_(idx, idx)]
        try:
            return np.trace(np.linalg.pinv(A_sub))
        except np.linalg.LinAlgError:
            return np.inf

    def solve(self):
        # Start with Infinity because we are MINIMIZING
        best_obj = np.inf
        best_idx = None

        # Try all combinations of size k
        for indices in itertools.combinations(range(self.p), self.k):
            # Calculate A-Optimality
            obj = self.calculate_obj(indices)

            # Update if we found a smaller (better) objective
            if obj < best_obj:
                best_obj = obj
                best_idx = list(indices)

        return np.array(best_idx), best_obj
