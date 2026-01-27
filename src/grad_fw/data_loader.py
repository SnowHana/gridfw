import numpy as np
import pandas as pd
import io
import requests
from ucimlrepo import fetch_ucirepo

# Registry of supported datasets

DATASETS_URL = {
    "residential": "https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx",
    "secom": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
    "arrhythmia": "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
}

DATASETS_ID = {"myocardial": 579}

DATASETS = DATASETS_URL | DATASETS_ID


def load_dataset_online(name_or_url):
    """
    General entry point to load any supported CSSP dataset.

    Args:
        name_or_url (str): Either a key ('residential', 'secom') or a direct URL.

    Returns:
        A (np.ndarray): Correlation matrix (p x p).
        X_norm (np.ndarray): Normalized feature matrix (N x p).
    """
    # 1. Resolve URL
    url = DATASETS_URL.get(name_or_url.lower(), name_or_url)

    print(f"  > Loading dataset: {name_or_url}")
    print(f"    Source: {url}")

    try:
        if name_or_url in DATASETS_URL:
            # 2. Download Data (Robust SSL handling)
            response = requests.get(url, verify=False)
            response.raise_for_status()
            content = io.BytesIO(response.content)

            # 3. Dispatch to specific loader based on known URLs
            if "Residential-Building" in url:
                X_raw = _parse_residential(content)
            elif "secom" in url:
                X_raw = _parse_secom(content)
            elif "arrhythmia" in url:
                X_raw = _parse_arrhythmia(content)
            else:
                # Fallback: Try generic CSV loading
                print("    Unknown format. Attempting generic CSV load...")
                df = pd.read_csv(content)
                X_raw = df.select_dtypes(include=[np.number]).to_numpy()

            if X_raw is None:
                return None, None
        elif name_or_url in DATASETS_ID:
            if "myocardial" in name_or_url:
                X = fetch_ucirepo(id=DATASETS_ID["myocardial"]).data.features
                X_raw = _parse_myocardial(X)
        # 4. Standardize and Compute Correlation (Shared Logic)
        return _standardize_and_correlate(X_raw)

    except Exception as e:
        print(f"    CRITICAL ERROR loading data: {e}")
        return None, None


# --- SPECIFIC PARSERS ---


def _parse_residential(content):
    """Parser for UCI Residential Building (Excel, Headers, targets at end)."""
    try:
        # Read Excel (Header is on row 1, index 1)
        df = pd.read_excel(content, header=1)

        # Columns 4 to 107 are the features (V-1 to V-104)
        # Drop first 4 (ID/Dates) and last 2 (Targets)
        X_df = df.iloc[:, 4:107]

        # Force numeric
        X_raw = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        return np.nan_to_num(X_raw)

    except Exception as e:
        print(f"    Error parsing Residential Excel: {e}")
        return None


def _parse_secom(content):
    """Parser for SECOM (Space-separated, No Header, Constant Columns)."""
    try:
        # Read CSV with space delimiter
        df = pd.read_csv(content, sep=r"\s+", header=None)
        X_raw = df.to_numpy(dtype=np.float64)

        # Fill NaNs (SECOM has many)
        X_raw = np.nan_to_num(X_raw)

        return _clean_constant_cols(X_raw, "secom")

    except Exception as e:
        print(f"    Error parsing SECOM CSV: {e}")
        return None


def _parse_arrhythmia(content):
    """Parser for Arrhythmia (Comma-separated, '?' for missing data)."""
    try:
        # FIX 1: Use sep="," (default) and handle '?' missing values
        df = pd.read_csv(content, header=None, na_values="?")

        # FIX 2: Arrhythmia often has a 'class' label in the last column
        # Usually for CSSP we only want the features (columns 0-278)
        X_df = df.iloc[:, :-1]

        # Force numeric and fill NaNs with 0
        X_raw = (
            X_df.apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )

        return _clean_constant_cols(X_raw, "arrhythmia")

    except Exception as e:
        print(f"    Error parsing Arrhythmia CSV: {e}")
        return None


def _parse_myocardial(content):
    try:
        X_raw = (
            content.apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )

        return _clean_constant_cols(X_raw, "myocardial")

    except Exception as e:
        print(f"    Error parsing Arrhythmia CSV: {e}")
        return None


# --- SHARED MATH ---


def _clean_constant_cols(X_raw, name):
    """Drop constant cols cuz constant cols will crash our normalisation"""
    std_devs = np.std(X_raw, axis=0)
    keep_idx = np.where(std_devs > 1e-9)[0]

    print(
        f"    [{name} Cleaning] Dropped {X_raw.shape[1] - len(keep_idx)} constant columns."
    )
    return X_raw[:, keep_idx]


def _standardize_and_correlate(X_raw):
    """
    Normalizes X (Z-score) and calculates A = (X^T X) / N.
    Used for ALL datasets to ensure consistent math.
    """
    N, p = X_raw.shape
    print(f"    Raw Data Shape: {N} rows x {p} features")

    # Z-score Normalization
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)

    # Safety: Avoid division by zero
    X_std[X_std == 0] = 1.0

    X_norm = (X_raw - X_mean) / X_std

    # Correlation Matrix
    A = (X_norm.T @ X_norm) / N

    print(f"    Computed Correlation Matrix A: {A.shape}")
    return A, X_norm


def get_correlation_density(A):
    """
    Docstring for get_correlation_density
    Return avg. correlation of off-diagonal to see how correlated it is
    :param A: Description
    """
    p = A.shape[0]
    # Extract only off-diagonal elements
    off_diag = A[~np.eye(p, dtype=bool)]
    return np.mean(np.abs(off_diag))


print(load_dataset_online("secom"))
