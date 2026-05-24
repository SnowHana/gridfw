import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import kagglehub
from grad_fw import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.benchmarks.BruteForceSolver import BruteForceSolver

# PATHS
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # examples/market -> repo root
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
os.makedirs(DATA_DIR, exist_ok=True)

KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
PRICES_CACHE = os.path.join(DATA_DIR, "sp500_prices.csv")
MAX_AGE_DAYS = 30

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"DATA_DIR:  {DATA_DIR}")


# Kaggle Data
def load_kaggle_data(force_update=False):
    companies_path = os.path.join(DATA_DIR, "sp500_companies.csv")
    if not force_update and os.path.exists(companies_path):
        age = (time.time() - os.path.getmtime(companies_path)) / 86400
        if age < MAX_AGE_DAYS:
            print(f"Using cached Kaggle data ({age:.0f} days old).")
        else:
            print(f"Kaggle cache is {age:.0f} days old, re-downloading...")
            kagglehub.dataset_download(KAGGLE_DATASET, output_dir=DATA_DIR)
    else:
        print("Downloading Kaggle dataset...")
        kagglehub.dataset_download(KAGGLE_DATASET, output_dir=DATA_DIR)

    return {
        name: pd.read_csv(os.path.join(DATA_DIR, f"sp500_{name}.csv"))
        for name in ["companies", "index"]
    }


# Price Data (cached)
def load_prices(tickers, start="2015-01-01", end="2024-12-31", force_update=False):
    if not force_update and os.path.exists(PRICES_CACHE):
        age = (time.time() - os.path.getmtime(PRICES_CACHE)) / 86400
        if age < MAX_AGE_DAYS:
            print(f"Loading cached prices ({age:.0f} days old)...")
            return pd.read_csv(PRICES_CACHE, index_col="Date", parse_dates=True)
        print(f"Price cache is {age:.0f} days old, re-downloading...")
    else:
        print("Downloading prices from yfinance (this takes ~2 min)...")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    raw = raw.dropna(axis=1, thresh=int(0.95 * len(raw)))
    raw = raw.ffill().dropna()
    raw.index.name = "Date"
    raw.to_csv(PRICES_CACHE)
    print(f"Saved {raw.shape[1]} stocks x {raw.shape[0]} days → {PRICES_CACHE}")
    return raw


# Return and Covariance matrix
def build_matrices(prices):
    X = np.log(prices / prices.shift(1)).dropna().values  # (T, p)
    A = np.cov(X.T)
    return X, A


dfs = load_kaggle_data()
tickers = dfs["companies"]["Symbol"].str.replace(".", "-", regex=False).tolist()

# 2. Load prices
prices = load_prices(tickers)
stock_names = prices.columns.tolist()
print(f"Prices shape: {prices.shape}  (days x stocks)")

X, A = build_matrices(prices=prices)
p = A.shape[0]
print(f"X: {X.shape} | A: {A.shape}")


# Run Solver
k = 10
total_variance = np.trace(A)

print(f"\nRunning FW-Homotopy (k={k}, p={p})...")
solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
s = solver.solve(verbose=True)
fw_indices = np.where(s > 0.5)[0]
fw_stocks = [stock_names[i] for i in fw_indices]
print(fw_stocks)

print(f"\nRunning Greedy...")
greedy_solver = GreedySolver(A, k)
greedy_s = greedy_solver.solve()[0]

# print(greedy_indices)
greedy_stocks = [stock_names[i] for i in greedy_s]
print(greedy_stocks)


print(f"\nRunning Brute Force...")
brute_solver = BruteForceSolver(A, k)
brute_s = brute_solver.solve()
brute_stocks = [stock_names[i] for i in brute_s]
print(brute_stocks)
# print(greedy_s)
# greedy_indices = np.where(greedy_s > 0.5)[0]
