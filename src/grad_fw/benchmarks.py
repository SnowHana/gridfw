import numpy as np
import itertools
import time

from grad_fw.fw_homotomy import FWHomotopySolver


class GreedyAOptSolver:
    """
    Standard Greedy Algorithm for A-Optimality (Minimizing Trace of Inverse).
    Selects one feature at a time to MINIMIZE the objective.
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def calculate_obj(self, indices):
        if len(indices) == 0:
            return np.inf
        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]
        try:
            return np.trace(np.linalg.inv(A_ss))
        except np.linalg.LinAlgError:
            try:
                return np.trace(np.linalg.pinv(A_ss))
            except np.linalg.LinAlgError:
                return np.inf

    def solve(self):
        start_time = time.time()
        current_indices = []
        for _ in range(self.k):
            best_idx = -1
            best_obj = np.inf
            candidates = [i for i in range(self.p) if i not in current_indices]
            for candidate in candidates:
                temp_indices = current_indices + [candidate]
                val = self.calculate_obj(temp_indices)
                if val < best_obj:
                    best_obj = val
                    best_idx = candidate
            if best_idx != -1:
                current_indices.append(best_idx)
        total_time = time.time() - start_time
        final_obj = self.calculate_obj(current_indices)
        return np.array(current_indices), final_obj, total_time


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
        Calculates Tr( A_SS^-1 (A^2)_SS ).
        Because Tr(XYZ) = Tr(YZX)
        """
        if len(indices) == 0:
            return 0.0

        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]  # A_ss = X_s^T @ X_s
        A2_ss = self.A2[np.ix_(idx, idx)]

        try:
            # Objective: MAXIMIZE Tr( A_SS^-1 A2_SS )
            # Try fast inverse first
            # Add small regularization to avoid singularity with inv
            inv_A_ss = np.linalg.inv(A_ss + 1e-6 * np.eye(len(idx)))
            return np.trace(inv_A_ss @ A2_ss)
        except np.linalg.LinAlgError:
            try:
                # Fallback to pseudo-inverse
                inv_A_ss = np.linalg.pinv(A_ss)
                return np.trace(inv_A_ss @ A2_ss)
            except np.linalg.LinAlgError:
                return 0.0

    def solve(self):
        start_time = time.time()
        current_indices = []

        # Greedily add k elements
        for _ in range(self.k):
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
            return np.trace(np.linalg.inv(A_sub))
        except np.linalg.LinAlgError:
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


class GreedyPortfolioSolver:
    """
    Greedy Algorithm for Minimum-Variance Portfolio.
    Equivalent to MAXIMIZING the sum of elements of the inverse covariance matrix.
    Objective: Maximize 1^T (A_S)^-1 1
    """

    def __init__(self, A, k):
        self.A = A
        self.p = A.shape[0]
        self.k = k

    def calculate_obj(self, indices):
        """
        Calculates 1^T (A_S)^-1 1.
        """
        if len(indices) == 0:
            return -np.inf  # We want to MAXIMIZE

        idx = list(indices)
        A_ss = self.A[np.ix_(idx, idx)]  # O(k^2) for k x k A_ss

        try:
            # Objective: MAXIMIZE sum of inverse elements
            # Add small regularization to avoid singularity with inv
            inv = np.linalg.inv(A_ss + 1e-6 * np.eye(len(idx)))  # O(k^3) for k x k A_ss
            return np.sum(inv)  # O(k^2)
        except np.linalg.LinAlgError:
            try:
                inv = np.linalg.pinv(A_ss)
                return np.sum(inv)
            except np.linalg.LinAlgError:
                return -np.inf

    def solve(self):
        start_time = time.time()
        current_indices = []

        # Greedily add k elements
        for _ in range(self.k):
            best_idx = -1
            best_obj = -np.inf

            candidates = [i for i in range(self.p) if i not in current_indices]

            for candidate in candidates:  # O(p)
                temp_indices = current_indices + [candidate]
                val = self.calculate_obj(temp_indices)  # O(k^3)  for k x k A_ss

                if val > best_obj:
                    best_obj = val
                    best_idx = candidate

            if best_idx != -1:
                current_indices.append(best_idx)
        # O(p * k^3)

        total_time = time.time() - start_time
        final_obj = self.calculate_obj(current_indices)  # O(k^3) for k x k A_ss
        # Overall O(p * k^3)
        return np.array(current_indices), final_obj, total_time


def run_experiment(
    A, k, steps, samples, experiment_name, alpha=0.01, dataset_name="Unknown"
):
    p = A.shape[0]
    print(
        f"\n--- {experiment_name} ({dataset_name}, p={p}, k={k}, steps={steps}, n_mc={samples}, alpha={alpha}) ---"
    )

    # 1. Greedy (Baseline)
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0

    # 2. FW-Homotopy
    solver = FWHomotopySolver(A, k, alpha=alpha, n_steps=steps, n_mc_samples=samples)
    t0 = time.time()
    s_fw = solver.solve(n_restarts=1, verbose=False)
    fw_time = time.time() - t0

    idx = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(idx))

    ratio = fw_obj / g_obj if g_obj != 0 else 0.0
    print(f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s")
    return {
        "experiment_name": experiment_name,
        "k": k,
        "steps": steps,
        "samples": samples,
        "g_obj": g_obj,
        "fw_obj": fw_obj,
        "g_time": g_time,
        "fw_time": fw_time,
        "dataset_name": dataset_name,
        "p": p,
        "ratio": ratio,
        "speedupx": g_time / fw_time,
    }
