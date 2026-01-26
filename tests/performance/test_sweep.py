import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver, GreedyPortfolioSolver
from grad_fw.data_loader import load_dataset_online
from grad_fw.benchmarks import run_experiment

# --- CONFIGURATION ---
# Add or remove dataset names here to control which datasets are tested in the sweeps.
DATASETS = ["secom"]

# --- HELPERS ---


# def run_experiment(
#     A, k, steps, samples, experiment_name, logger, alpha=0.01, dataset_name="Unknown"
# ):
#     p = A.shape[0]
#     print(
#         f"\n--- {experiment_name} ({dataset_name}, p={p}, k={k}, steps={steps}, n_mc={samples}, alpha={alpha}) ---"
#     )

#     # 1. Greedy (Baseline)
#     greedy = GreedySolver(A, k)
#     t0 = time.time()
#     _, g_obj, _ = greedy.solve()
#     g_time = time.time() - t0

#     # 2. FW-Homotopy
#     solver = FWHomotopySolver(A, k, alpha=alpha, n_steps=steps, n_mc_samples=samples)
#     t0 = time.time()
#     s_fw = solver.solve(n_restarts=1, verbose=False)
#     fw_time = time.time() - t0

#     idx = np.where(s_fw > 0.5)[0]
#     fw_obj = greedy.calculate_obj(list(idx))

#     ratio = fw_obj / g_obj if g_obj != 0 else 0.0
#     print(f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s")

#     logger(
#         experiment_name,
#         k,
#         steps,
#         samples,
#         g_obj,
#         fw_obj,
#         g_time,
#         fw_time,
#         dataset_name=dataset_name,
#         p=p,
#     )
#     return ratio


# def run_portfolio_experiment(A, k, steps, samples, experiment_name, logger, dataset_name="Unknown"):
#     p = A.shape[0]
#     print(f"\n--- {experiment_name} (Portfolio) ({dataset_name}, p={p}, k={k}, steps={steps}) ---")

#     # 1. Greedy Portfolio (Maximization)
#     greedy = GreedyPortfolioSolver(A, k)
#     t0 = time.time()
#     _, g_obj, _ = greedy.solve()
#     g_time = time.time() - t0

#     # 2. FW-Homotopy Portfolio
#     solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=steps, n_mc_samples=samples, objective_type='portfolio')
#     t0 = time.time()
#     s_fw = solver.solve(n_restarts=1, verbose=False)
#     fw_time = time.time() - t0

#     idx = np.where(s_fw > 0.5)[0]
#     fw_obj = greedy.calculate_obj(list(idx))

#     ratio = fw_obj / g_obj if g_obj != 0 else 0.0
#     print(f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s")

#     logger(
#         experiment_name,
#         k,
#         steps,
#         samples,
#         g_obj,
#         fw_obj,
#         g_time,
#         fw_time,
#         dataset_name=dataset_name,
#         p=p
#     )
#     return ratio

# --- FIXTURES ---


# --- TESTS ---


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_sweep_vary_k(dataset_data, sweep_logger):
    """Experiment 1: Vary k with fixed steps/samples."""
    A, name = dataset_data
    # Optimized range: 10 to 150 with step 15 (~10 points for good plotting)
    k_values = range(10, 450, 25)

    steps = 800
    samples = 100

    for k in k_values:
        res = run_experiment(A, k, steps, samples, "vary_k", dataset_name=name)
        sweep_logger(**res)


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_sweep_vary_nmc(dataset_data, sweep_logger):
    """Experiment 2: Vary n_mc at k=10."""
    A, name = dataset_data
    k = 10
    steps = 800
    nmc_values = range(10, 500, 35)

    for nmc in nmc_values:
        res = run_experiment(A, k, steps, nmc, "vary_nmc", dataset_name=name)
        sweep_logger(**res)


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_sweep_vary_steps(dataset_data, sweep_logger):
    """Experiment 3: Vary steps at k=20."""
    A, name = dataset_data
    k = 20
    samples = 100
    step_values = range(100, 3000, 50)

    for s in step_values:
        res = run_experiment(A, k, s, samples, "vary_steps", dataset_name=name)
        sweep_logger(**res)


def test_sweep_vary_p(sweep_logger):
    """Experiment 8: Vary Dimension p (Synthetic Data)."""
    p_values = range(25, 700, 10)
    k = 20
    steps = 800
    samples = 100

    np.random.seed(42)

    for p in p_values:
        # Generate synthetic data for each p
        X = np.random.randn(1000, p)
        A = X.T @ X

        res = run_experiment(
            A, k, steps, samples, "vary_p", dataset_name="Synthetic_VaryP"
        )
        sweep_logger(**res)


# @pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
# def test_sweep_portfolio_vary_k(dataset_data, sweep_logger):
#     """Experiment 4: Portfolio Optimization - Vary k."""
#     A, name = dataset_data
#     k_values = [10, 20, 30, 40, 50]
#     steps = 800
#     samples = 1

#     for k in k_values:
#         run_portfolio_experiment(A, k, steps, samples, "portfolio_vary_k", sweep_logger, dataset_name=name)


@pytest.mark.parametrize("dataset_data", ["synthetic"], indirect=True)
def test_sweep_large_synthetic(dataset_data, sweep_logger):
    """Experiment 5: Large Synthetic Problem (p=500, k=50)."""
    A, name = dataset_data
    k = 50
    steps = 1000
    samples = 50
    res = run_experiment(
        A, k, steps, samples, "large_synthetic", dataset_name=name
    )
    sweep_logger(**res)


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_sweep_tuning(dataset_data, sweep_logger):
    """Experiment 7: Tuning (Grid Search over Steps and Samples)."""
    A, name = dataset_data
    k = 50  # Fixed k for tuning

    # Grid Search Space
    step_values = [200, 500, 800, 1200]
    sample_values = [10, 30, 50, 100]

    for s in step_values:
        for nmc in sample_values:
            res = run_experiment(
                A, k, s, nmc, f"tuning_s{s}_n{nmc}", dataset_name=name
            )
            sweep_logger(**res)


@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_sweep_vary_alpha(dataset_data, sweep_logger):
    """Experiment 6: Vary Alpha (Homotopy Step Size)."""
    A, name = dataset_data
    k = 20
    steps = 1000
    samples = 100
    alpha_values = [0.001, 0.01, 0.05, 0.1]

    for alpha in alpha_values:
        res = run_experiment(
            A,
            k,
            steps,
            samples,
            f"vary_alpha_{alpha}",
            alpha=alpha,
            dataset_name=name,
        )
        sweep_logger(**res)
