import numpy as np
import time

from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.fw_homotomy import FWHomotopySolver


def run_experiment(
    A,
    k,
    experiment_name,
    alpha=0.01,
    dataset_name="Unknown",
    steps=None,
    samples=None,
):
    """run_experiment _summary_

    Args:
        A (_type_): Pre-processed Dataset
        k (_type_): Column Subset size
        experiment_name (_type_):
        alpha (float, optional): Step size. Defaults to 0.01.
        dataset_name (str, optional): . Defaults to "Unknown".
        steps (_type_, optional): n (Step number). Defaults to None.
        samples (_type_, optional): m (Rademacher sample size). Defaults to None.

    Returns:
        dict: Result dictionary
    """
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

    # 3. Return Result
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
    """find_critical_k : Binary Search to find k_c : speedupx = 1.0 (if isTime) or ratio = 1.0 (if not isTime)

    Args:
        A_sub (_type_): Dataset (Sub)matrix
        name_p (_type_): Experiment Name
        logger (_type_): Logger
        max_run (int, optional): Max iteration for binary search. Defaults to 10.
        steps (_type_, optional): n. Defaults to None.
        samples (_type_, optional): m. Defaults to None.
        isTime (bool, optional): True: Speedupx (1.0), False: Ratio (1.0). Defaults to True(Speedupx).
        target (float, optional): target speedupx/ratio value. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
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

    # Result
    best_res = {}
    best_diff = float("inf")

    run_name = f"{name_p}_p{p}"

    # Binary Search
    for i in range(max_run):
        k = (low + high) // 2
        if k < 1:
            k = 1
        if k > p:
            k = p

        # Run exp
        res_dict = run_experiment(
            A_sub,
            k,
            steps=min(20 * k, 1000),
            samples=samples,
            experiment_name=experiment_name,
            dataset_name=run_name,
        )

        # NOTE: Run EXP with original setting
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
