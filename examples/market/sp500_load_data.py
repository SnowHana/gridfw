import os
import time
import pandas as pd
import yfinance as yf
import kagglehub

# PATHS
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # examples/market -> repo root
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
PRICES_CACHE = os.path.join(DATA_DIR, "sp500_prices.csv")
MAX_AGE_DAYS = 30

# Dates
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"


def download_kaggle_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    kagglehub.dataset_download(KAGGLE_DATASET, output_dir=DATA_DIR)


def load_sp500_companies_index_df():
    """Read CSV files, load sp500 companies name(ticker) and sp500 index by dates

    Returns:
        _type_: _description_
    """
    res = {}
    for value in ["companies", "index"]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"sp500_{value}.csv"))
        res[value] = df

    return res


# Kaggle Data
def load_kaggle_data(force_update=False):
    """load_kaggle_data : Load a dataset for sp500 companies name and sp500 index
    Use Cached dataset if not too old, else download new dataset from Kaggle

    Args:
        force_update (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    companies_path = os.path.join(DATA_DIR, "sp500_companies.csv")
    if not force_update and os.path.exists(companies_path):
        cached_data_age = (time.time() - os.path.getmtime(companies_path)) / 86400
        if cached_data_age < MAX_AGE_DAYS:
            # Use Cached Dataset
            print(f"Using cached Kaggle data ({cached_data_age:.0f} days old).")
        else:
            # Cache outdated, new dataset
            print(f"Kaggle cache is {cached_data_age:.0f} days old, re-downloading...")
            download_kaggle_dataset()
    else:
        # Force update or first run (file doesn't exist yet)
        print("Downloading Kaggle dataset...")
        download_kaggle_dataset()

    return load_sp500_companies_index_df()


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
    dfs = load_kaggle_data()
    tickers = dfs["companies"]["Symbol"].str.replace(".", "-", regex=False).tolist()

    # 2. Load prices
    prices_df = load_prices(tickers, start, end)
    stock_names = prices_df.columns.tolist()
    # print(f"Prices shape: {prices_df.shape}  (days x stocks)")
    return prices_df


# load_sp500_dataset()
