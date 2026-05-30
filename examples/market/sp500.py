import numpy as np
from grad_fw import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver
from sp500_load_data import load_sp500_prices_df

# START_DATE = "2025-01-01"
# END_DATE = "2026-01-01"


# Return and Covariance matrix
def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna().values


def compute_covariance(X):
    return np.cov(X.T)


def solve_sp500_fw_homotopy(A, k, stock_names):
    # print(f"\nRunning FW-Homotopy (k={k}, p={A.shape[0]})...")
    solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
    solution = solver.solve()
    fw_indices = FWHomotopySolver.selected_indices(solution)
    fw_stocks = sorted([stock_names[i] for i in fw_indices])
    # print(fw_stocks)
    return fw_indices


def solve_sp500_greedy(A, k, stock_names):
    # print(f"\nRunning Greedy...")
    greedy_solver = GreedySolver(A, k)
    greedy_s = greedy_solver.solve()[0]

    greedy_stocks = sorted([stock_names[i] for i in greedy_s])
    # print(greedy_stocks)
    return greedy_s


