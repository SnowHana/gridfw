import pytest
import csv
import os
import sys
from datetime import datetime

LOG_FILE = "benchmark_log.csv"
SESSION_RESULTS = []  # Store results here for the terminal summary


def pytest_addoption(parser):
    parser.addoption("--msg", action="store", default="", help="Custom note for log")


@pytest.fixture(scope="session")
def benchmark_logger(request):
    user_note = request.config.getoption("--msg")

    # 1. Create CSV if missing
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Dataset",
                    "Test_Name",
                    "k",
                    "Steps",
                    "Samples",
                    "Greedy_Obj",
                    "FW_Obj",
                    "Ratio",
                    "Greedy_Time_s",
                    "FW_Time_s",
                    "Speedup_x",
                    "Status",
                    "Note",
                ]
            )

    def log_result(
        test_name,
        k,
        steps,
        samples,
        g_obj=None,
        fw_obj=None,
        g_time=None,
        fw_time=None,
        status="PASS",
    ):

        # Calculate Derived Metrics
        ratio = (fw_obj / g_obj) if (fw_obj and g_obj) else 0.0
        # Only calc speedup if we have valid times
        speedup = (g_time / fw_time) if (g_time and fw_time and fw_time > 0) else 0.0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. Add to Global Session List (For the Terminal Table)
        SESSION_RESULTS.append(
            {
                "name": test_name,
                "k": k,
                "ratio": ratio,
                "speedup": speedup,
                "status": status,
            }
        )

        # 3. Write to CSV (Persistent Log)
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    "SECOM",
                    test_name,
                    k,
                    steps,
                    samples,
                    f"{g_obj:.4f}" if g_obj else "",
                    f"{fw_obj:.4f}" if fw_obj else "",
                    f"{ratio:.4f}" if ratio else "",
                    f"{g_time:.4f}" if g_time else "",  # Will be empty if None
                    f"{fw_time:.4f}" if fw_time else "",
                    f"{speedup:.2f}" if speedup else "",
                    status,
                    user_note,
                ]
            )

    return log_result


# --- NEW: The Hook that runs after all tests finish ---
def pytest_sessionfinish(session, exitstatus):
    """
    Prints a pretty summary table to the console after tests complete.
    """
    if not SESSION_RESULTS:
        return

    # Print Header
    print("\n" + "=" * 65)
    print(f"{'BENCHMARK SUMMARY':^65}")
    print("=" * 65)
    print(
        f"{'Test Name':<25} | {'k':<4} | {'Ratio':<8} | {'Speedup':<8} | {'Status':<6}"
    )
    print("-" * 65)

    # Print Rows
    for r in SESSION_RESULTS:
        # Format metrics nicely
        ratio_str = f"{r['ratio']:.1%}" if r["ratio"] > 0 else "N/A"
        speed_str = f"{r['speedup']:.1f}x" if r["speedup"] > 0 else "N/A"

        # Colorize Status (optional simple ASCII indicator)
        status_str = r["status"]

        print(
            f"{r['name']:<25} | {r['k']:<4} | {ratio_str:<8} | {speed_str:<8} | {status_str:<6}"
        )

    print("=" * 65 + "\n")
