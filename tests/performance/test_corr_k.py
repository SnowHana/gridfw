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
def test_parameter_tuning(dataset_data, sweep_logger):
    """Test 1: Vary k with fixed steps, samples, p"""
    A, name = dataset_data
    p = A.shape[0]

    fixed_k = int(p * 0.1)
    # n_mc_values = [20]
    n_mc_values = [20, 50, 100, 200]
    # alpha_values = [0.001]
    alpha_values = [0.001, 0.01, 0.1, 0.5]

    for n_mc in n_mc_values:
        for alpha in alpha_values:
            exp_name = f"tune_a{alpha}_nmc{n_mc}"

            res = run_experiment(
                A,
                fixed_k,
                experiment_name=exp_name,
                dataset_name=name,
                samples=n_mc,
                alpha=alpha,
            )
            sweep_logger(**res)
