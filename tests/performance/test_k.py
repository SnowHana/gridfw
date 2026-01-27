"""
Test effect of k (Subset size) in terms of time spent, and accuracy
"""

import pytest
import numpy as np
from grad_fw.benchmarks import run_experiment, find_critical_k
from grad_fw.data_loader import DatasetLoader

# DATASETS_URL = []
# DATASETS_URL = ["residential", "arrhythmia"]

DATASETS = [
    "synthetic_high_corr",
    "mnist",
    "madelon",
    "synthetic_toeplitz",
    "residential",
    "secom",
    "arrhythmia",
    "myocardial",
]
# DATASETS = ["residential"]


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_high_corr_k(dataset_data, sweep_logger):
    """Test 1: Vary k with fixed steps, samples, p"""
    A, name = dataset_data
    p = A.shape[0]
    gap = max(1, int(0.01 * p))
    # Test 0.01 * p, 0.05 * p, 0.1 * p

    k_values = [gap * i for i in range(1, 2)]

    for k in k_values:
        res = run_experiment(A, k, "dense_k", dataset_name=name)
        sweep_logger(**res)


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
