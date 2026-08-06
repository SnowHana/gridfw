import pandas as pd
import numpy as np
import numpy.ma as ma
import os
from grad_fw import FWHomotopySolver

# PATHS
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
    # repo root
)

DATA_DIR = os.path.join(REPO_ROOT, "data", "market")


def comput_log_returns(prices_df: pd.DataFrame):
    return np.log(prices_df / prices_df.shift(1)).dropna().values


def compute_covariances(X: pd.DataFrame):
    return np.cov(X.T)  # We want num_Stocks x num_Stocks shaped


def _load_covariance(universe_id: str):
    """Returns (A, tickers) , covariance matrix + list of ticker symbols in same order as A's rows/columns

    Args:
        universe_id (str): ie) "sp500" for sp500
    """
    df = pd.read_csv(
        os.path.join(DATA_DIR, f"{universe_id}_prices.csv"),
        index_col="Date",
        parse_dates=True,
    )

    log_return = comput_log_returns(df)

    cov = compute_covariances(log_return)

    return cov, df.columns.tolist()


def _load_sector_map(universe_id: str) -> dict[str, str]:
    """Returns {ticker : sector} for every stock in sp500_companies.csv

    Returns:
        dict[str, str]: _description_
    """
    df = pd.read_csv(os.path.join(DATA_DIR, f"{universe_id}_companies.csv"))
    return dict(zip(df.Symbol, df.Sector))


def _select_stocks(A, tickers, k):
    """Runs FW-Homotopy CSSP on A, returns list of dicts:
    [{"ticker": ..., "weight": ...}, ...] for the k selected stocks

    Args:
        A (_type_): _description_
        tickers (_type_): _description_
        k (_type_): _description_
    """

    solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
    solution = solver.solve(verbose=False)
    fw_tickers = ma.masked_where(solution < 0.5, tickers).compressed()
    return [{"ticker": str(t), "weight": 1 / k} for t in fw_tickers]


if __name__ == "__main__":
    _load_covariance("sp500")
