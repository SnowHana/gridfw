import json
import os

import numpy as np
import pandas as pd

from grad_fw import FWHomotopySolver

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "precomputed")

REG_EPS = 1e-6  # matches FWHomotopySolver's Tikhonov regularization

UNIVERSES = {
    "sp500": {
        "id": "sp500",
        "label": "S&P 500",
        "k_options": [10, 20, 30, 40, 50],
        "description": "S&P 500 constituents, 2018-2026 daily returns",
    }
}

_covariance_cache: dict[str, tuple[np.ndarray, list[str]]] = {}


class UnknownUniverseError(ValueError):
    pass


class UnsupportedKError(ValueError):
    pass


def list_universes() -> list[dict]:
    result = []
    for universe_id, meta in UNIVERSES.items():
        _, tickers = _get_covariance(universe_id)
        result.append(
            {
                "id": meta["id"],
                "label": meta["label"],
                "n_stocks": len(tickers),
                "k_options": meta["k_options"],
                "description": meta["description"],
            }
        )
    return result


def get_replication(universe_id: str, k: int, force_recompute: bool = False) -> dict:
    """Cache-or-compute: serve a cached JSON result if present, else run
    FW-Homotopy once and cache the result for next time."""
    if universe_id not in UNIVERSES:
        raise UnknownUniverseError(f"Unknown universe: {universe_id!r}")

    k_options = UNIVERSES[universe_id]["k_options"]
    if k not in k_options:
        raise UnsupportedKError(
            f"k={k} is not supported for {universe_id!r}. Options: {k_options}"
        )

    cache_path = _cache_path(universe_id, k)
    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path) as f:
            result = json.load(f)
            result["precomputed"] = True
            return result

    result = _compute_replication(universe_id, k)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _cache_path(universe_id: str, k: int) -> str:
    return os.path.join(CACHE_DIR, f"{universe_id}_k{k}.json")


def _load_companies() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "sp500_companies.csv"))


def _sector_map() -> dict[str, str]:
    companies = _load_companies()
    tickers = companies["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(tickers, companies["Sector"]))


def _get_covariance(universe_id: str) -> tuple[np.ndarray, list[str]]:
    if universe_id != "sp500":
        raise UnknownUniverseError(f"Unknown universe: {universe_id!r}")

    if universe_id in _covariance_cache:
        return _covariance_cache[universe_id]

    prices = pd.read_csv(
        os.path.join(DATA_DIR, "sp500_prices.csv"), index_col="Date", parse_dates=True
    )
    log_returns = np.log(prices / prices.shift(1)).dropna()
    tickers = log_returns.columns.tolist()
    A = np.cov(log_returns.values.T)

    _covariance_cache[universe_id] = (A, tickers)
    return A, tickers


def _compute_replication(universe_id: str, k: int) -> dict:
    A, tickers = _get_covariance(universe_id)
    p = A.shape[0]

    solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
    solution = solver.solve(verbose=False)
    idx = sorted(FWHomotopySolver.selected_indices(solution).tolist())

    A_reg = A + REG_EPS * np.eye(p)
    A2_reg = A_reg @ A_reg
    idx_arr = np.array(idx)
    A_ss = A_reg[np.ix_(idx_arr, idx_arr)]
    A2_ss = A2_reg[np.ix_(idx_arr, idx_arr)]

    cssp_objective = float(np.trace(np.linalg.inv(A_ss) @ A2_ss))
    full_objective = float(np.trace(A_reg))  # Tr(A^-1 A^2) == Tr(A) exactly
    coverage_pct = round(cssp_objective / full_objective * 100, 2)

    sector_map = _sector_map()
    weight = round(1.0 / len(idx), 4)
    selected = [
        {
            "ticker": tickers[i],
            "sector": sector_map.get(tickers[i], "Unknown"),
            "weight": weight,
        }
        for i in idx
    ]

    return {
        "universe": universe_id,
        "k": k,
        "selected": selected,
        "cssp_objective": round(cssp_objective, 4),
        "coverage_pct": coverage_pct,
        "precomputed": True,
    }
