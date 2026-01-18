import numpy as np
import itertools
from grad_verif.core import ProblemGenerator
from grad_fw.fw_homotomy import FWHomotopySolver


class BruteForceSolver:
    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def calculate_objective(self, indices):
        """Calculates Trace( (A_ss)^-1 A_s: A_:s )"""
        if len(indices) != self.k:
            return -1.0
        A_ss = self.A[np.ix_(indices, indices)]
        A_s_all = self.A[indices, :]
        try:
            # Objective: Trace( (A_ss)^-1 @ (A_s_all @ A_s_all.T) )
            return np.trace(np.linalg.inv(A_ss) @ (A_s_all @ A_s_all.T))
        except np.linalg.LinAlgError:
            return -1.0

    def solve(self):
        best_obj = -np.inf
        best_indices = None
        # Check all combinations
        for indices in itertools.combinations(range(self.p), self.k):
            obj = self.calculate_objective(list(indices))
            if obj > best_obj:
                best_obj = obj
                best_indices = list(indices)
        return best_indices, best_obj
