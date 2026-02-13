import numpy as np


import itertools


class BruteForceSolver:
    """
    Exact solver that tries ALL combinations to find the global MINIMUM.
    Complexity: O(p choose k)
    """

    def __init__(self, A, k):
        self.A = A
        self.k = k
        self.p = A.shape[0]

    def calculate_obj(self, indices):
        idx = list(indices)
        A_sub = self.A[np.ix_(idx, idx)]
        try:
            return np.trace(np.linalg.inv(A_sub))
        except np.linalg.LinAlgError:
            try:
                return np.trace(np.linalg.pinv(A_sub))
            except np.linalg.LinAlgError:
                return np.inf

    def solve(self):
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
