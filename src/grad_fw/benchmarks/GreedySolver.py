import numpy as np


import time


class GreedySolver:
    """
    Greedy Algorithm for CSSP (Maximizing Trace of Projection).
    Objective: Maximize Tr( X^T P_S X ) = Tr( A_SS^-1 (A^2)_SS )
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

        # Precompute A^2 for efficiency
        self.A2 = A @ A

    def calculate_obj(self, indices):
        """
        Calcualtes objective value for selected indices (=s)
        Calculates Tr( A_SS^-1 (A^2)_SS ).
        Because Tr(XYZ) = Tr(YZX), refer to doc for further proof.
        """
        if len(indices) == 0:
            return 0.0

        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]  # A_ss = X_s^T @ X_s
        A2_ss = self.A2[np.ix_(idx, idx)]

        try:
            # Objective: MAXIMIZE Tr( A_SS^-1 A2_SS )
            # Try fast inverse first
            # Tikhonov Regularization
            inv_A_ss = np.linalg.inv(A_ss + 1e-9 * np.eye(len(idx)))
            return np.trace(inv_A_ss @ A2_ss)
        except np.linalg.LinAlgError:
            try:
                # Pseudo Inverse for singular...
                inv_A_ss = np.linalg.pinv(A_ss)
                return np.trace(inv_A_ss @ A2_ss)
            except np.linalg.LinAlgError:
                # Can't inverse, return -inf
                return -np.inf

    def solve(self):
        """
        Docstring for solve
        Choose 1 best index (column) that improves our objective value the most
        Forward Selection
        """
        start_time = time.time()
        current_indices = []

        # Greedily add k elements
        for _ in range(self.k):
            # Choose 1 best column amongst UNSELECTED
            best_idx = -1
            best_obj = -np.inf

            candidates = [i for i in range(self.p) if i not in current_indices]

            for candidate in candidates:
                temp_indices = current_indices + [candidate]
                val = self.calculate_obj(temp_indices)

                if val > best_obj:
                    best_obj = val
                    best_idx = candidate

            if best_idx != -1:
                current_indices.append(best_idx)

        total_time = time.time() - start_time
        final_obj = self.calculate_obj(current_indices)

        return np.array(current_indices), final_obj, total_time
