"""
Test effect of k (Subset size) in terms of time spent, and accuracy
"""

import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver, run_experiment
from grad_fw.data_loader import load_dataset_online, DATASETS_URL

# DATASETS_URL = []
DATASETS_URL = ["residential", "arrhythmia"]


@pytest.mark.parametrize("dataset_data", DATASETS_URL, indirect=True)
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
    p = A_sub.shape[0]

    if isTime:
        experiment_name = "critical_k_time"
        col_type = "speedupx"
    else:
        experiment_name = "critical_k_accuracy"
        col_type = "ratio"
    # Binary search
    low = 1
    high = p
    best_k = -1
    closest_diff = float("inf")
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
        res = res_dict[col_type]

        # Log critical k specific data
        log_data = res_dict.copy()
        log_data["critical_k"] = k
        log_data[col_type] = res
        log_data["p"] = p  # Ensure p is logged
        logger(**log_data)

        # Exact match: 95% ~ 105%
        if 0.95 <= res <= 1.05:
            return {"k": k, col_type: res}

        # Track best k
        if abs(res - 1.0) <= closest_diff:
            closest_diff = abs(res - 1.0)
            best_k = k

        # Binary search
        if res > 1.0:
            # Reduce k
            high = k - 1
        else:
            low = k + 1

        # Exhausted feasible cases
        if low > high:
            return {"k": best_k, col_type: res}


# @pytest.mark.parametrize("partition", 30)
test_critical_k_data = [(i, 30) for i in range(5)]


@pytest.mark.parametrize(
    "run_id, num_points", test_critical_k_data
)  # Run test 5 times and log
@pytest.mark.parametrize("dataset_data", DATASETS_URL, indirect=True)
def test_critical_k(
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
        results[p] = res["k"]

        # Log final critical k
        critical_k_final_logger(
            dataset_name=name, p=p, final_critical_k=res["k"], speedup=res["speedx"]
        )

    print(results)
    return
