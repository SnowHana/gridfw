import numpy as np
import itertools


class BruteForceSolver:
    """
    Exact solver that tries ALL combinations to find the global MAXIMUM.
    Objective: Maximize Tr( A_ss^-1 (A^2)_ss ) = Tr( X^T P_s X )
    Complexity: O(p choose k)
    """

    def __init__(self, A, k):
        self.A = A
        self.k = k
        self.p = A.shape[0]
        self.A2 = A @ A  # Precompute full A^2

    def calculate_obj(self, indices):
        idx = list(indices)
        A_sub = self.A[np.ix_(idx, idx)]
        A2_sub = self.A2[np.ix_(idx, idx)]

        try:
            return np.trace(np.linalg.inv(A_sub + 1e-9 * np.eye(len(idx))) @ A2_sub)
        except np.linalg.LinAlgError:
            try:
                return np.trace(np.linalg.pinv(A_sub) @ A2_sub)
            except np.linalg.LinAlgError:
                return -np.inf

    def solve(self):
        best_obj = -np.inf
        best_idx = None

        for indices in itertools.combinations(range(self.p), self.k):
            obj = self.calculate_obj(indices)
            if obj > best_obj:
                best_obj = obj
                best_idx = list(indices)

        return np.array(best_idx), best_obj
