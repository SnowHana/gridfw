import numpy as np
import pandas as pd
import io
import requests
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml  # Added for MNIST/Madelon


class DatasetLoader:
    """
    Unified loader for CSSP benchmarks.
    Handles downloading, parsing, cleaning, and standardizing datasets.
    Supported: 'residential', 'secom', 'arrhythmia', 'myocardial', 'mnist', 'madelon', 'synthetic'.
    """

    # Registry of supported datasets
    DATASETS_URL = {
        "residential": "https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx",
        "secom": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
        "arrhythmia": "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
    }

    DATASETS_ID = {"myocardial": 579}

    def __init__(self):
        pass

    def load(self, name_or_url: str, **kwargs):
        """
        Main entry point to load a dataset.

        Args:
            name_or_url: Name ('mnist', 'synthetic') or URL.
            **kwargs: Arguments passed to synthetic generator (N, p, correlation_strength).

        Returns:
            A (np.ndarray): Correlation matrix (p x p).
            X_norm (np.ndarray): Normalized feature matrix (N x p).
        """
        key = name_or_url.lower()
        print(f"  > Loading dataset: {key}")

        try:
            # --- PATH 1: Synthetic Generator ---
            if key == "synthetic":
                return self.generate_high_dim_correlated_data(**kwargs)

            # --- PATH 2: Scikit-Learn OpenML (MNIST, Madelon) ---
            if key in ["mnist", "madelon"]:
                return self._load_openml(key)

            # --- PATH 3: UCI ML Repo ID ---
            if key in self.DATASETS_ID:
                print(f"    Source: UCI Repo ID {self.DATASETS_ID[key]}")
                X_data = fetch_ucirepo(id=self.DATASETS_ID[key]).data.features
                X_raw = self._parse_myocardial(X_data)
                return self._standardize_and_correlate(X_raw)

            # --- PATH 4: Direct URL / Legacy ---
            url = self.DATASETS_URL.get(key, name_or_url)
            print(f"    Source: {url}")

            response = requests.get(url, verify=False)
            response.raise_for_status()
            content = io.BytesIO(response.content)

            X_raw = None
            if "Residential-Building" in url:
                X_raw = self._parse_residential(content)
            elif "secom" in url:
                X_raw = self._parse_secom(content)
            elif "arrhythmia" in url:
                X_raw = self._parse_arrhythmia(content)
            else:
                # Generic Fallback
                print("    Unknown format. Attempting generic CSV load...")
                df = pd.read_csv(content)
                X_raw = df.select_dtypes(include=[np.number]).to_numpy()

            if X_raw is None:
                return None, None

            return self._standardize_and_correlate(X_raw)

        except Exception as e:
            print(f"    CRITICAL ERROR loading data: {e}")
            return None, None

    # --- OPENML HANDLER ---
    def _load_openml(self, key):
        """Handles MNIST and Madelon via sklearn."""
        if key == "mnist":
            print("    Fetching MNIST from OpenML (this may take a moment)...")
            # Load MNIST (70k samples). We subsample to 2000 for solver speed.
            X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
            X = X[:2000].astype(np.float64)  # Subsample first 2000

        elif key == "madelon":
            print("    Fetching Madelon from OpenML...")
            X, _ = fetch_openml("madelon", version=1, return_X_y=True, as_frame=False)
            X = X.astype(np.float64)

        # Clean constant columns (Vital for MNIST border pixels)
        X = self._clean_constant_cols(X, key)
        return self._standardize_and_correlate(X)

    # --- SYNTHETIC GENERATOR ---
    def generate_high_dim_correlated_data(
        self, N=2000, p=1000, n_blocks=20, correlation_strength=0.9
    ):
        """Generates synthetic 'Trap' data for Greedy vs FW testing."""
        # Calculate block size ensuring integer division
        block_size = p // n_blocks
        actual_p = block_size * n_blocks

        X = np.zeros((N, actual_p))
        print(
            f"    Generating Synthetic Data: N={N}, p={actual_p}, Blocks={n_blocks}, Corr={correlation_strength}"
        )

        for i in range(n_blocks):
            # Latent Factor (Hidden Truth)
            latent_factor = np.random.randn(N)

            # Generate noisy versions of latent factor
            start_col = i * block_size
            end_col = (i + 1) * block_size

            noise = np.random.randn(N, block_size)
            # Mix signal and noise
            X[:, start_col:end_col] = (
                correlation_strength * latent_factor[:, None]
                + (1 - correlation_strength) * noise
            )

        # Standardize directly (skip _standardize_and_correlate to avoid double print/logic)
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
        A = (X_norm.T @ X_norm) / N

        print(f"    Computed Correlation Matrix A: {A.shape}")
        return A, X_norm

    # --- PARSERS (Unchanged) ---
    def _parse_residential(self, content):
        try:
            df = pd.read_excel(content, header=1)
            X_df = df.iloc[:, 4:107]
            X_raw = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float64
            )
            return np.nan_to_num(X_raw)
        except Exception as e:
            print(f"    Error parsing Residential: {e}")
            return None

    def _parse_secom(self, content):
        try:
            df = pd.read_csv(content, sep=r"\s+", header=None)
            X_raw = np.nan_to_num(df.to_numpy(dtype=np.float64))
            return self._clean_constant_cols(X_raw, "secom")
        except Exception as e:
            print(f"    Error parsing SECOM: {e}")
            return None

    def _parse_arrhythmia(self, content):
        try:
            df = pd.read_csv(content, header=None, na_values="?")
            X_df = df.iloc[:, :-1]
            X_raw = (
                X_df.apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )
            return self._clean_constant_cols(X_raw, "arrhythmia")
        except Exception as e:
            print(f"    Error parsing Arrhythmia: {e}")
            return None

    def _parse_myocardial(self, X_df):
        try:
            X_raw = (
                X_df.apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )
            return self._clean_constant_cols(X_raw, "myocardial")
        except Exception as e:
            print(f"    Error parsing Myocardial: {e}")
            return None

    # --- UTILS ---
    def _clean_constant_cols(self, X_raw, name):
        """Drops columns with variance ~0."""
        std_devs = np.std(X_raw, axis=0)
        keep_idx = np.where(std_devs > 1e-9)[0]
        dropped_count = X_raw.shape[1] - len(keep_idx)

        if dropped_count > 0:
            print(f"    [{name} Cleaning] Dropped {dropped_count} constant columns.")

        return X_raw[:, keep_idx]

    def _standardize_and_correlate(self, X_raw):
        """Z-score normalization and A = (X^T X)/N."""
        N, p = X_raw.shape
        print(f"    Raw Data Shape: {N} rows x {p} features")

        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0)
        X_std[X_std == 0] = 1.0  # Safety

        X_norm = (X_raw - X_mean) / X_std
        A = (X_norm.T @ X_norm) / N

        print(f"    Computed Correlation Matrix A: {A.shape}")
        return A, X_norm

    @staticmethod
    def get_correlation_density(A):
        p = A.shape[0]
        off_diag = A[~np.eye(p, dtype=bool)]
        return np.mean(np.abs(off_diag))
