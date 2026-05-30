import os
import time
import pandas as pd
import yfinance as yf

# PATHS
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # examples/market -> repo root
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
MAX_AGE_DAYS = 30

# Dates
PRICES_CACHE = os.path.join(DATA_DIR, "sp500_prices.csv")
# PRICES_FILE_NAME
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"


def load_sp500_companies_index_df():
    """Load static S&P 500 company metadata from committed CSV files.

    To update: manually download from https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
    and replace data/market/sp500_companies.csv and sp500_index.csv.
    """
    res = {}
    for value in ["companies", "index"]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"sp500_{value}.csv"))
        res[value] = df
    return res


# Price Data (cached)


def clean_raw_price_data(raw_prices_df):
    raw_prices_df = raw_prices_df.dropna(axis=1, thresh=int(0.95 * len(raw_prices_df)))
    raw_prices_df = raw_prices_df.ffill().dropna()
    raw_prices_df.index.name = "Date"
    raw_prices_df.to_csv(PRICES_CACHE)
    print(
        f"Saved {raw_prices_df.shape[1]} stocks x {raw_prices_df.shape[0]} days → {PRICES_CACHE}"
    )
    return raw_prices_df


def download_yf_prices_data(tickers, start, end):
    print("Downloading prices from yfinance (this takes ~2 min)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)[
        "Adj Close"
    ]  # Use 'Adj Close'
    return clean_raw_price_data(raw_prices_df=raw)


def check_cached_prices_data(force_update):
    """Return cached DataFrame if fresh, else None (caller must re-download)."""
    if not force_update and os.path.exists(PRICES_CACHE):
        age = (time.time() - os.path.getmtime(PRICES_CACHE)) / 86400
        if age < MAX_AGE_DAYS:
            print(f"Loading cached prices ({age:.0f} days old)...")
            return pd.read_csv(PRICES_CACHE, index_col="Date", parse_dates=True)
        print(f"Price cache is {age:.0f} days old, will re-download.")
        return None
    return None


def load_prices(tickers, start="2015-01-01", end="2026-03-30", force_update=False):
    """load_prices Load Prices data using cached data / yfinacne

    Args:
        tickers (_type_): _description_
        start (str, optional): _description_. Defaults to "2015-01-01".
        end (str, optional): _description_. Defaults to "2024-12-31".
        force_update (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    cached_data = check_cached_prices_data(force_update)

    if cached_data is not None:
        return cached_data
    else:
        return download_yf_prices_data(tickers, start, end)


# Usage
def load_sp500_prices_df(start=START_DATE, end=END_DATE):
    """Load S&P 500 company metadata and price history.

    Returns:
        dict with keys:
            prices_df:      DataFrame (days × stocks) of adjusted close prices
    """
    dfs = load_sp500_companies_index_df()
    tickers = dfs["companies"]["Symbol"].str.replace(".", "-", regex=False).tolist()

    # 2. Load prices
    prices_df = load_prices(tickers, start, end)
    return prices_df
