import pytest
import numpy as np
from grad_fw.benchmarks.benchmarks import run_experiment, find_critical_k
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
@pytest.mark.parametrize("n_mc", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
def test_parameter_tuning(dataset_data, n_mc, alpha, sweep_logger):
    """
    Test 1: Grid Search for Alpha and NMC.
    Pytest will automatically generate a unique test for every combination.
    """
    A, name = dataset_data
    p = A.shape[0]

    # Calculate k once per dataset
    fixed_k = max(1, int(p * 0.1))

    # Create a unique name for the log
    exp_name = f"tune_a{alpha}_nmc{n_mc}"

    # Run the single experiment
    # Note: 'steps' is None so it defaults to dynamic logic (20 * k) inside the class
    res = run_experiment(
        A,
        fixed_k,
        experiment_name=exp_name,
        dataset_name=name,
        samples=n_mc,
        alpha=alpha,
    )

    # Log it
    sweep_logger(**res)
