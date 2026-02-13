"""
Test effect of k (Subset size) in terms of time spent, and accuracy
"""

import pytest
import numpy as np
from grad_fw.benchmarks.benchmarks import run_experiment
from grad_fw.data_loader import (
    ALL_DATASETS,
    DATASETS_SYNTHETIC,
    DATASETS_URL,
    DATASETS_ID,
    DATASETS_OPENML,
)

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
# DATASETS = ["myocardial"]
# DATASETS = {**DATASETS_URL, **DATASETS_ID, **DATASETS_OPENML}
# DATASETS = DATASETS_SYNTHETIC


@pytest.mark.parametrize("m", [500])
@pytest.mark.parametrize("alpha", [0.2])
@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_compare_k_fixed_p(dataset_data, compare_k_logger, alpha, m):
    """Test 1: Tetsting objective values and speedupx for alpha = 0.1, 0.2 and m = 20, 50
    Those were results that tended to be ideal from previous test"""
    A, name = dataset_data
    p = A.shape[0]
    k_list = [int(0.01 * p), int(0.1 * p), int(0.2 * p), int(0.3 * p), int(0.5 * p)]
    for k in k_list:
        res = run_experiment(
            A=A,
            k=k,
            experiment_name=f"compare_k{k}_a{alpha}_m{m}_block_1",
            dataset_name=name,
            samples=m,
            alpha=alpha,
        )
        compare_k_logger(**res)
