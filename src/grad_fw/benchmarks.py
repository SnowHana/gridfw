import numpy as np
import itertools
import time

from grad_fw.fw_homotomy import FWHomotopySolver


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
                # Fallback to pseudo-inverse
                # A bit slower, but better when A_ss is messy (singular)
                inv_A_ss = np.linalg.pinv(A_ss)
                return np.trace(inv_A_ss @ A2_ss)
            except np.linalg.LinAlgError:
                return 0.0

    def solve(self):
        """
        Docstring for solve
        Choose 1 best index (column) that improves our objective value the most
        :param self: Description
        """
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


def run_experiment(
    A,
    k,
    experiment_name,
    alpha=0.01,
    dataset_name="Unknown",
    steps=None,
    samples=None,
):
    p = A.shape[0]

    # 1. Greedy (Baseline)
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0

    # 2. FW-Homotopy
    solver = FWHomotopySolver(A, k, alpha=alpha, n_steps=steps, n_mc_samples=samples)

    t0 = time.time()
    s_fw = solver.solve(verbose=False)
    fw_time = time.time() - t0

    idx = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(idx))

    ratio = fw_obj / g_obj if g_obj != 0 else 0.0

    # ---  Retrieve the ACTUAL values used by the solver ---
    actual_steps = solver.n_steps
    actual_samples = solver.n_mc_samples

    print(
        f"--- {experiment_name} ({dataset_name}, k={k}) ---\n"
        f"Used Steps: {actual_steps} | Used NMC: {actual_samples} | Alpha: {alpha}\n"
        f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s"
    )

    return {
        "experiment_name": experiment_name,
        "k": k,
        "steps": actual_steps,
        "samples": actual_samples,
        "alpha": alpha,
        "g_obj": g_obj,
        "fw_obj": fw_obj,
        "g_time": g_time,
        "fw_time": fw_time,
        "dataset_name": dataset_name,
        "p": p,
        "ratio": ratio,
        "speedupx": g_time / fw_time if fw_time > 0 else 0,
    }


def find_critical_k(
    A_sub, name_p, logger, max_run=10, steps=None, samples=None, isTime=True, target=1.0
):
    """Binary Search to find k_c : speedupx = 1.0 or ratio = 1.0
    isTime decides"""
    p = A_sub.shape[0]

    if isTime:
        experiment_name = "critical_k_time"
        target_col = "speedupx"
    else:
        experiment_name = "critical_k_accuracy"
        target_col = "ratio"
    # Binary search
    low = 1
    high = p

    # Return dictionary
    best_res = {}
    best_diff = float("inf")

    run_name = f"{name_p}_p{p}"
    for i in range(max_run):
        k = (low + high) // 2

        if k < 1:
            k = 1
        if k > p:
            k = p
        # Run exp (No adaptive)
        res_dict = run_experiment(
            A_sub,
            k,
            steps=min(20*k, 1000),
            samples=samples,
            experiment_name=experiment_name,
            dataset_name=run_name,
        )
        # Run EXP with original setting
        # res_dict = run_experiment(
        #     A_sub,
        #     k,
        #     steps=steps,
        #     samples=samples,
        #     experiment_name=experiment_name,
        #     dataset_name=run_name,
        # )
        res = res_dict[target_col]

        # Log critical k specific data
        log_data = res_dict.copy()
        log_data["critical_k"] = k
        log_data["p"] = p  # Ensure p is logged
        logger(**log_data)

        # Track best k
        if abs(res - target) <= best_diff:
            best_diff = abs(res - target)
            best_res = res_dict.copy()
            best_res["k"] = k

        # Exact match: 95% ~ 105%
        if target - 0.05 <= res <= target + 0.05:
            return best_res

        # Binary search
        if res > target:
            # Reduce k
            high = k - 1
        else:
            low = k + 1

        # Exhausted feasible cases
        if low > high:
            break
    return best_res
