import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver, GreedyPortfolioSolver
from grad_fw.data_loader import load_dataset


@pytest.fixture(scope="module")
def residential_data():
    A, _ = load_dataset("residential")
    if A is None:
        pytest.skip("Could not load Residential data")
    return A


@pytest.fixture(scope="module")
def secom_data():
    A, _ = load_dataset("secom")
    if A is None:
        pytest.skip("Could not load secom data")
    return A


def run_experiment(A, k, steps, samples, experiment_name, logger, alpha=0.01):
    print(f"\n--- {experiment_name} (k={k}, steps={steps}, n_mc={samples}, alpha={alpha}) ---")
    
    # 1. Greedy (Baseline)
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0

    # 2. FW-Homotopy
    solver = FWHomotopySolver(A, k, alpha=alpha, n_steps=steps, n_mc_samples=samples)
    t0 = time.time()
    s_fw = solver.solve(n_restarts=1, verbose=False)
    fw_time = time.time() - t0

    idx = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(idx))

    ratio = fw_obj / g_obj
    print(f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s")

    logger(
        experiment_name,
        k,
        steps,
        samples,
        g_obj,
        fw_obj,
        g_time,
        fw_time
    )
    return ratio


def test_sweep_vary_k(residential_data, sweep_logger):
    """Experiment 1: Vary k with fixed steps/samples."""
    k_values = [10, 20, 30, 40, 50, 60]
    steps = 800
    samples = 100

    for k in k_values:
        run_experiment(
            residential_data, 
            k, 
            steps, 
            samples, 
            "vary_k", 
            sweep_logger
        )



def test_secom_sweep_vary_k(secom_data, sweep_logger):
    """Experiment 2: Vary k with fixed steps/samples on secom data."""
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    steps = 800
    samples = 100

    for k in k_values:
        run_experiment(
            secom_data, 
            k, 
            steps, 
            samples, 
            "vary_k", 
            sweep_logger
        )

def test_sweep_vary_nmc(residential_data, sweep_logger):
    """Experiment 2: Vary n_mc at k=10 (problematic case)."""
    k = 10
    steps = 800
    nmc_values = [50, 100, 300, 500]

    for nmc in nmc_values:
        run_experiment(
            residential_data, 
            k, 
            steps, 
            nmc, 
            "vary_nmc", 
            sweep_logger
        )


def test_sweep_vary_steps(residential_data, sweep_logger):
    """Experiment 3: Vary steps at k=20."""
    k = 20
    samples = 100
    step_values = [500, 1000, 2000]

    for s in step_values:
        run_experiment(
            residential_data, 
            k, 
            s, 
            samples, 
            "vary_steps", 
            sweep_logger
        )


# def run_portfolio_experiment(A, k, steps, samples, experiment_name, logger):
#     print(f"\n--- {experiment_name} (Portfolio) (k={k}, steps={steps}) ---")
    
#     # 1. Greedy Portfolio (Maximization)
#     greedy = GreedySolver(A, k)
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

#     ratio = fw_obj / g_obj
#     print(f"Ratio: {ratio:.4f} | Time: {fw_time:.4f}s")

#     logger(
#         experiment_name,
#         k,
#         steps,
#         samples,
#         g_obj,
#         fw_obj,
#         g_time,
#         fw_time
#     )
#     return ratio


# def test_sweep_portfolio_vary_k(residential_data, sweep_logger):
#     """Experiment 4: Portfolio Optimization (Maximization) - Vary k."""
#     k_values = [10, 20, 30, 40, 50]
#     steps = 800
#     samples = 1  # Deterministic objective, samples not used but passed

#     for k in k_values:
#         run_portfolio_experiment(
#             residential_data, 
#             k, 
#             steps, 
#             samples, 
#             "portfolio_vary_k", 
#             sweep_logger
#         )


def test_sweep_large_synthetic(sweep_logger):
    """Experiment 5: Large Synthetic Problem (p=500, k=50)."""
    p = 500
    k = 50
    steps = 1000
    samples = 50
    
    print(f"\n=== Large Synthetic Problem (p={p}, k={k}) ===")
    np.random.seed(42)
    X = np.random.randn(1000, p)
    A = X.T @ X
    
    run_experiment(A, k, steps, samples, "large_synthetic", sweep_logger)


def test_sweep_vary_alpha(residential_data, sweep_logger):
    """Experiment 6: Vary Alpha (Homotopy Step Size)."""
    k = 20
    steps = 1000
    samples = 100
    alpha_values = [0.001, 0.01, 0.05, 0.1]

    for alpha in alpha_values:
        run_experiment(
            residential_data, 
            k, 
            steps, 
            samples, 
            f"vary_alpha_{alpha}", 
            sweep_logger,
            alpha=alpha
        )
