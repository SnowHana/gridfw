import numpy as np
import pytest
import csv
import os
import sys
from datetime import datetime

from grad_fw.data_loader import DatasetLoader

LOG_FILE = "logs/benchmark_log.csv"
SESSION_RESULTS = []  # Store results here for the terminal summary

LOADER = DatasetLoader()


def pytest_addoption(parser):
    parser.addoption("--msg", action="store", default="", help="Custom note for log")


@pytest.fixture
def dataset_data(request):
    name = request.param
    # Standard loader usage
    A, _ = LOADER.load(name)
    if A is None:
        pytest.skip(f"Could not load {name} data")
    return A, name.capitalize()


SWEEP_LOG_FILE = "logs/param_sweep_log.csv"
GRAD_LOG_FILE = "logs/grad_test_log.csv"
CRITICAL_K_LOG_FILE = "logs/critical_k_results.csv"


class CSVLogger:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, **data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp]
        for h in self.headers[1:]:
            # Map result dict keys to header names
            # Logic for mapping: lowercase header, replace spaces/hyphens with underscores
            key = h.lower().replace(" ", "_").replace("-", "_")

            # --- MAPPINGS ---
            if key == "speedup_x":
                key = "speedupx"
            elif key == "greedy_obj":
                key = "g_obj"
            elif key == "fw_obj":
                key = "fw_obj"
            elif key == "greedy_time_s":
                key = "g_time"
            elif key == "fw_time_s":
                key = "fw_time"
            elif key == "test_name":
                key = "experiment_name"
            elif key == "experiment":
                key = "experiment_name"
            elif key == "dataset":
                key = "dataset_name"
            elif key == "speedup_at_k":
                key = "speedupx"
            elif key == "final_critical_k":
                key = "final_critical_k"
            # Explicit mapping for alpha is usually not needed if key matches 'alpha',
            # but we leave it standard logic.

            val = data.get(key, "")
            # Formatting
            if isinstance(val, (float, np.float64, np.float32)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


@pytest.fixture(scope="session")
def benchmark_logger(request):
    user_note = request.config.getoption("--msg")
    headers = [
        "Timestamp",
        "Dataset",
        "Test_Name",
        "k",
        "Steps",
        "Samples",
        "Alpha",  # <--- ADDED
        "Greedy_Obj",
        "FW_Obj",
        "Ratio",
        "Greedy_Time_s",
        "FW_Time_s",
        "Speedup_x",
        "Status",
        "Note",
    ]
    logger = CSVLogger(LOG_FILE, headers)

    def log_result(**kwargs):
        # Calculate derived metrics if not present
        if "ratio" not in kwargs and kwargs.get("g_obj") and kwargs.get("fw_obj"):
            kwargs["ratio"] = kwargs["fw_obj"] / kwargs["g_obj"]
        if "speedupx" not in kwargs and kwargs.get("g_time") and kwargs.get("fw_time"):
            kwargs["speedupx"] = (
                kwargs["g_time"] / kwargs["fw_time"] if kwargs["fw_time"] > 0 else 0
            )

        kwargs.setdefault("status", "PASS")
        kwargs.setdefault("note", user_note)
        kwargs.setdefault("alpha", "")  # Default to empty if not provided

        # Add to Global Session List for summary
        SESSION_RESULTS.append(
            {
                "name": kwargs.get("experiment_name")
                or kwargs.get("test_name")
                or "Unknown",
                "k": kwargs.get("k", 0),
                "ratio": kwargs.get("ratio", 0),
                "speedup": kwargs.get("speedupx", 0),
                "status": kwargs.get("status", "PASS"),
            }
        )
        logger.log(**kwargs)

    return log_result


@pytest.fixture(scope="session")
def sweep_logger():
    headers = [
        "Timestamp",
        "Dataset",
        "p",
        "Experiment",
        "k",
        "Steps",
        "Samples",
        "Alpha",  # <--- ADDED
        "Greedy_Obj",
        "FW_Obj",
        "Ratio",
        "Greedy_Time_s",
        "FW_Time_s",
        "Speedup_x",
        "Status",
    ]
    logger = CSVLogger(SWEEP_LOG_FILE, headers)

    def log_sweep(**kwargs):
        kwargs.setdefault("status", "DONE")
        kwargs.setdefault("alpha", "")  # Default if missing
        logger.log(**kwargs)

    return log_sweep


@pytest.fixture(scope="session")
def critical_k_logger():
    headers = ["Timestamp", "Dataset", "p", "Critical_k", "Speedup_At_k", "Ratio"]
    logger = CSVLogger(CRITICAL_K_LOG_FILE, headers)

    def log_critical(**kwargs):
        logger.log(**kwargs)

    return log_critical


CRITICAL_K_FINAL_LOG_FILE = "logs/critical_k_final.csv"


@pytest.fixture(scope="session")
def critical_k_final_logger():
    headers = ["Timestamp", "Dataset", "p", "Final_Critical_k", "Speedup_At_k", "Ratio"]
    logger = CSVLogger(CRITICAL_K_FINAL_LOG_FILE, headers)

    def log_critical_final(**kwargs):
        logger.log(**kwargs)

    return log_critical_final


CRITICLAL_K_NMC_LOG_FILE = "logs/critical_k_nmc.csv"


@pytest.fixture(scope="session")
def critical_k_nmc_logger():
    headers = [
        "Timestamp",
        "Dataset",
        "p",
        "Final_Critical_k",
        "Samples",
        "Steps",
        "Speedup_At_k",
        "Ratio",
    ]
    logger = CSVLogger(CRITICLAL_K_NMC_LOG_FILE, headers)

    def log_critical_k_nmc(**kwargs):
        logger.log(**kwargs)

    return log_critical_k_nmc


# --- NEW: Automatic Logging for Gradient Tests ---
def pytest_runtest_logreport(report):
    """
    Automatically logs results of tests in 'tests/grad/' to a CSV.
    """
    if report.when == "call":
        # Check if the test is in the 'grad' directory
        if "tests/grad/" in report.nodeid:
            os.makedirs("logs", exist_ok=True)

            # Initialize CSV if missing
            if not os.path.exists(GRAD_LOG_FILE):
                with open(GRAD_LOG_FILE, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Test_Name", "Outcome", "Duration_s"])

            # Log the result
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(GRAD_LOG_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        timestamp,
                        report.nodeid,
                        report.outcome.upper(),
                        f"{report.duration:.4f}",
                    ]
                )
