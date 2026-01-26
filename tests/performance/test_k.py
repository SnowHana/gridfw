"""
Test effect of k (Subset size) in terms of time spent, and accuracy
"""

import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver, run_experiment
from grad_fw.data_loader import load_dataset_online, DATASETS

# DATASETS_URL = []
# DATASETS_URL = ["residential", "arrhythmia"]


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
@pytest.mark.parametrize("num_k", [10])
def test_dense_k(dataset_data, sweep_logger, num_k):
    """Test 1: Vary k with fixed steps, samples, p"""
    A, name = dataset_data
    gap = A.shape[0] // num_k
    k_list = range(gap, gap * (num_k - 1), gap)

    steps = 800
    samples = 100

    for k in k_list:
        res = run_experiment(A, k, steps, samples, "dense_k", dataset_name=name)
        sweep_logger(**res)


def find_critical_k(
    A_sub, name_p, logger, max_run=10, steps=800, samples=100, isTime=True
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
        # Run exp
        res_dict = run_experiment(
            A_sub,
            k,
            steps=steps,
            samples=samples,
            experiment_name=experiment_name,
            dataset_name=run_name,
        )
        res = res_dict[target_col]

        # Log critical k specific data
        log_data = res_dict.copy()
        log_data["critical_k"] = k
        log_data["p"] = p  # Ensure p is logged
        logger(**log_data)

        # Track best k
        if abs(res - 1.0) <= best_diff:
            best_diff = abs(res - 1.0)
            best_res = res_dict.copy()
            best_res["k"] = k

        # Exact match: 95% ~ 105%
        if 0.95 <= res <= 1.05:
            return best_res

        # Binary search
        if res > 1.0:
            # Reduce k
            high = k - 1
        else:
            low = k + 1

        # Exhausted feasible cases
        if low > high:
            break
    return best_res


# @pytest.mark.parametrize("partition", 30)
test_critical_k_data = [(i, 30) for i in range(5)]


@pytest.mark.parametrize(
    "run_id, num_points", test_critical_k_data
)  # Run test 5 times and log
@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_critical_k_time(
    dataset_data, critical_k_logger, critical_k_final_logger, run_id, num_points
):
    """Test 2: Binary search to find k value such that speedup is close to 1"""
    A_full, name = dataset_data
    p_full = A_full.shape[0]
    p_values = np.linspace(25, p_full, num=num_points, dtype=int)
    p_values = np.unique(p_values)
    # p_values = range(50, p_full, 25)
    results = {}
    for p in p_values:
        # Sub-matrix
        indices = np.random.choice(p_full, p, replace=False)

        A_sub = A_full[np.ix_(indices, indices)]
        res = find_critical_k(
            A_sub=A_sub, name_p=name, logger=critical_k_logger, max_run=10, isTime=True
        )

        best_k = res.get("k", -1)
        results[p] = best_k

        # Log final critical k
        critical_k_final_logger(
            dataset_name=name,
            p=p,
            final_critical_k=best_k,
            speedup=res["speedupx"],
            ratio=res.get("ratio"),
        )

    print(results)
    return


@pytest.mark.parametrize(
    "run_id, num_points", test_critical_k_data
)  # Run test 5 times and log
@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_critical_k_accuracy(
    dataset_data, critical_k_logger, critical_k_final_logger, run_id, num_points
):
    """Test 2: Binary search to find k value such that speedup is close to 1"""
    A_full, name = dataset_data
    p_full = A_full.shape[0]
    p_values = np.linspace(25, p_full, num=num_points, dtype=int)
    p_values = np.unique(p_values)
    # p_values = range(50, p_full, 25)
    results = {}
    for p in p_values:
        # Sub-matrix
        indices = np.random.choice(p_full, p, replace=False)

        A_sub = A_full[np.ix_(indices, indices)]
        res = find_critical_k(
            A_sub=A_sub, name_p=name, logger=critical_k_logger, max_run=10, isTime=False
        )
        best_k = res.get("k", -1)
        results[p] = best_k

        # Log final critical k
        critical_k_final_logger(
            dataset_name=name,
            p=p,
            final_critical_k=best_k,
            speedup=res["speedupx"],
            ratio=res.get("ratio"),
        )

    print(results)
    return
