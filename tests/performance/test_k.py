import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver, run_experiment
from grad_fw.data_loader import load_dataset

DATASETS = ["secom"]


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
        run_experiment(A, k, steps, samples, "dense_k", sweep_logger, dataset_name=name)
    # p_list = range(A.shape[0] // 10, A.shape[0] + 1, A.shape[0] // 10)


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_critical_k(dataset_data, sweep_logger, max_run=10):
    """Test 2: Test around boundary of k"""
    A, name = dataset_data
    p = A.shape[0]

    # Binary search
    low = 1
    high = p
    best_k = -1
    closest_diff = float("inf")

    for i in range(max_run):
        k = (low + high) // 2

        if k < 1:
            k = 1
        if k > p:
            k = p
        # Run exp
        res = run_experiment(
            A,
            k,
            steps=800,
            samples=100,
            experiment_name="critical_k",
            logger=sweep_logger,
            dataset_name=name,
        )["speedupx"]

        if 0.95 <= res <= 1.05:
            return
        if abs(res - 1.0) <= closest_diff:
            closest_diff = abs(res - 1.0)
            best_k = k

        # Binary search
        if res > 1.0:
            # Reduce k
            high = k - 1
        else:
            low = k + 1

        if low > high:
            return
