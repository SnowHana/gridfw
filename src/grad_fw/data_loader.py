import numpy as np
import pandas as pd
import io
import requests


def load_residential_building_data():
    """
    Fetches the Residential Building Data Set (ID: 437) directly via HTTP.
    Returns the Correlation Matrix (A) and normalized data (X_norm).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx"
    print(f"  > Downloading data directly from: {url}")

    try:
        # 1. Download the Excel file
        # verify=False prevents SSL errors on Mac if certificates are tricky
        response = requests.get(url, verify=False)
        response.raise_for_status()

        # 2. Read Excel
        # The file has two header rows. We read from row 1 (index 1) to get proper labels.
        df = pd.read_excel(io.BytesIO(response.content), header=1)

        # 3. Clean Data
        # The dataset structure is:
        # Cols 0-3: Date/ID info (Drop)
        # Cols 4-107: Features (Keep)
        # Cols 108-109: Targets (Drop)

        # Select numeric features only (Columns 4 to 107)
        X_df = df.iloc[:, 4:107]

        # Convert to numpy and ensure numeric type
        X_raw = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

        # Fill any NaNs with 0
        if np.isnan(X_raw).any():
            X_raw = np.nan_to_num(X_raw)

        print(f"    Loaded Raw Data: {X_raw.shape} (Rows x Features)")

        # 4. Standardize (Z-score normalization)
        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0)

        # Prevent division by zero if a column is constant
        X_std[X_std == 0] = 1.0

        X_norm = (X_raw - X_mean) / X_std

        # 5. Compute Correlation Matrix A = (X^T X) / N
        N = X_norm.shape[0]
        A = (X_norm.T @ X_norm) / N

        print(f"    Computed Correlation Matrix A: {A.shape}")
        return A, X_norm

    except Exception as e:
        print(f"    CRITICAL ERROR loading data: {e}")
        return None, None
