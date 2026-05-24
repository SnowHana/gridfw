import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import kagglehub

# PATHS
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
os.makedirs(DATA_DIR, exist_ok=True)

KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
PRICES_CACHE = os.path.join(DATA_DIR, "sp500_prices.csv")
MAX_AGE_DAYS = 30


# Kaggle Data
def load_kaggle_data(force_update=False):
    """Download S&P 500 metadata from Kaggle if missing or stale."""
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


# Price Data
def load_prices(tickers, start="2015-01-01", end="2024-12-31", force_update=False):
    """Load adjusted close prices, downloading from yfinance only when needed."""
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
    print(f"Saved {raw.shape[1]} stocks x {raw.shape[0]} days to cache.")
    return raw


# Return and Covariance matrix
def build_matrices(prices):
    """Compute demeaned log-return matrix X and covariance matrix A = X^T X."""
    X = np.log(prices / prices.shift(1)).dropna().values  # (T, p)
    A = np.cov(X.T)
    return X, A


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from grad_fw import FWHomotopySolver
    from grad_fw.benchmarks.GreedySolver import GreedySolver

    # 1. Load metadata
    dfs = load_kaggle_data()
    tickers = dfs["companies"]["Symbol"].str.replace(".", "-", regex=False).tolist()

    # 2. Load prices
    prices = load_prices(tickers)
    stock_names = prices.columns.tolist()
    print(f"Prices shape: {prices.shape}  (days x stocks)")

    # 3. Build matrices
    X, A = build_matrices(prices)
    p = A.shape[0]
    print(f"X: {X.shape} | A: {A.shape}")

    # 4. Run solvers
    k = 20
    total_variance = np.trace(A)

    print(f"\nRunning FW-Homotopy (k={k}, p={p})...")
    solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
    s = solver.solve(verbose=True)
    fw_indices = np.where(s > 0.5)[0]
    fw_stocks = [stock_names[i] for i in fw_indices]

    print(f"\nRunning Greedy baseline...")
    greedy = GreedySolver(A, k)
    greedy_indices, g_obj, g_time = greedy.solve()
    greedy_stocks = [stock_names[i] for i in greedy_indices]

    fw_obj = greedy.calculate_obj(list(fw_indices))

    # 5. Results
    print(f"\n=== Results (k={k} from p={p}) ===")
    print(f"Total variance (Tr A): {total_variance:.2f}")
    print(
        f"Greedy:  obj={g_obj:.2f}  ({g_obj/total_variance*100:.1f}% variance explained)  time={g_time:.2f}s"
    )
    print(
        f"FW:      obj={fw_obj:.2f}  ({fw_obj/total_variance*100:.1f}% variance explained)"
    )
    print(f"\nFW selected:     {fw_stocks}")
    print(f"Greedy selected: {greedy_stocks}")
