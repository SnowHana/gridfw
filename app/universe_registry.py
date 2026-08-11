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

UNIVERSES = {
    "sp500": {
        "label": "S&P 500",
        "k_min": 2,
        "description": "S&P 500 constituents, 2018-2026 daily returns",
    }
}


class UnknownUniverseError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class UnsupportedKError(ValueError):
    """Exception from not supported k

    Args:
        ValueError (_type_): _description_
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def comput_log_returns(prices_df: pd.DataFrame):
    return np.log(prices_df / prices_df.shift(1)).dropna().values


def compute_covariances(X: pd.DataFrame):
    return np.cov(X.T)  # We want num_Stocks x num_Stocks shaped


def _load_tickers(universe_id: str) -> list[str]:
    """Cheaply returns just the ticker list, without computing covariance.

    Reads only the CSV header (nrows=0) - avoids the O(p^3)-ish covariance
    calc when all we need is the stock count/names (e.g. for validating k or
    listing universe metadata).
    """
    header = pd.read_csv(
        os.path.join(DATA_DIR, f"{universe_id}_prices.csv"), index_col="Date", nrows=0
    )
    return header.columns.tolist()


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
    """_summary_

    Args:
        A (_type_): _description_
        tickers (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """

    solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
    solution = solver.solve(verbose=False)
    idx = np.where(solution > 0.5)[0]
    selected = [{"ticker": str(tickers[i]), "weight": 1 / k} for i in idx]

    cssp_objective = solver._calculate_obj(solution)
    full_objective = solver._calculate_obj(np.ones(len(tickers)))
    coverage_pct = cssp_objective / full_objective * 100

    return selected, cssp_objective, coverage_pct


def get_replication(universe_id: str, k: int) -> dict:
    """Returns a dict matching the ReplicationResponse schema

    Args:
        universe_id (str): _description_
        k (int): _description_

    Returns:
        dict: _description_
    """
    # Check universe, k
    if universe_id not in UNIVERSES:
        raise UnknownUniverseError(f"Unknown universe: {universe_id!r}")

    k_min = UNIVERSES[universe_id]["k_min"]
    k_max = len(_load_tickers(universe_id))
    if not (k_min <= k <= k_max):
        raise UnsupportedKError(
            f"k={k} is out of range for {universe_id}: must be between {k_min} and {k_max}"
        )

    A, tickers = _load_covariance(universe_id=universe_id)
    sector_map = _load_sector_map(universe_id=universe_id)
    selected_tickers, cssp_obj, coverge_pct = _select_stocks(A=A, tickers=tickers, k=k)

    # Add sector key...
    for s in selected_tickers:
        s["sector"] = sector_map[s["ticker"]]
    return {
        "universe": universe_id,
        "k": k,
        "selected": selected_tickers,
        "cssp_objective": cssp_obj,
        "coverage_pct": coverge_pct,
        "precomputed": False,
    }


def list_universes() -> list[dict]:
    """Returns a list of dicts matching UniverseInfo, one per entry in UNIVERSES

    Returns:
        list[dict]: _description_
    """
    res = []
    for universe_id, universe_data in UNIVERSES.items():
        n_stocks = len(_load_tickers(universe_id))
        res.append(
            {
                "id": universe_id,
                "label": universe_data["label"],
                "n_stocks": n_stocks,
                "k_min": universe_data["k_min"],
                "k_max": n_stocks,
                "description": universe_data["description"],
            }
        )

    return res


if __name__ == "__main__":
    _load_covariance("sp500")
