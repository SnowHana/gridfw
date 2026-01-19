import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver
from grad_fw.data_loader import load_dataset


# --- FIXTURE ---
@pytest.fixture(scope="module")
def secom_data():
    """Loads SECOM dataset once."""
    print("\n[Setup] Loading SECOM Data...")
    A, _ = load_dataset("secom")
    if A is None:
        pytest.skip("Could not load SECOM data")
    return A


# --- HELPER: The Benchmark Engine ---
def run_comparison(test_name, A, k, steps, samples, logger, threshold=0.90):
    """
    Standardized runner for comparing Greedy vs FW-Homotopy.
    Captures Time, Objective, Ratio, and Speedup.
    """
    print(f"\n=== {test_name} (k={k}) ===")

    # 1. Run Greedy (Baseline)
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0

    # 2. Run FW-Homotopy (Challenger)
    solver = FWHomotopySolver(A, k, n_steps=steps, n_mc_samples=samples)
    t0 = time.time()
    s_fw = solver.solve(verbose=False)
    fw_time = time.time() - t0

    # 3. Evaluate Metrics
    fw_indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(fw_indices))

    ratio = fw_obj / g_obj if g_obj != 0 else 0
    speedup = g_time / fw_time if fw_time > 0 else 0

    print(f"Greedy: {g_time:.4f}s | Obj: {g_obj:.4f}")
    print(f"FW:     {fw_time:.4f}s | Obj: {fw_obj:.4f}")
    print(f"Stats:  Ratio={ratio:.2%} | Speedup={speedup:.2f}x")

    # 4. Log Result
    status = "PASS" if ratio > threshold else "FAIL"
    logger(
        test_name,
        k,
        steps,
        samples,
        g_obj=g_obj,
        fw_obj=fw_obj,
        g_time=g_time,
        fw_time=fw_time,
        status=status,
    )

    return ratio, speedup


# --- TESTS ---


def test_secom_accuracy_k10(secom_data, benchmark_logger):
    """
    Small k=10. Focus: High Accuracy (Ratio).
    """
    ratio, _ = run_comparison(
        "accuracy_k10",
        secom_data,
        k=10,
        steps=500,
        samples=50,
        logger=benchmark_logger,
    )
    assert ratio > 0.90, f"Accuracy poor. Ratio: {ratio:.2%}"


def test_secom_benchmark_k50(secom_data, benchmark_logger):
    """
    Medium k=50. Focus: Speedup & Accuracy.
    """
    ratio, speedup = run_comparison(
        "benchmark_k50",
        secom_data,
        k=50,
        steps=800,
        samples=30,
        logger=benchmark_logger,
    )
    assert ratio > 0.90, "Accuracy dropped below 90%"
    assert speedup > 1.0, "FW should be faster than Greedy at k=50"


def test_secom_stress_k100(secom_data, benchmark_logger):
    """
    Large k=100. Focus: Extreme Speedup.
    """
    ratio, speedup = run_comparison(
        "stress_k100",
        secom_data,
        k=100,
        steps=500,
        samples=50,
        logger=benchmark_logger,
        threshold=0.85,  # Slightly looser threshold for hard stress test
    )
    # Check Accuracy
    assert ratio > 0.85, "Accuracy too low on stress test"
    # Check Speed (Should be significantly faster)
    assert speedup > 2.0, f"Speedup insufficient! Got {speedup:.2f}x"


def test_secom_stability(secom_data, benchmark_logger):
    """
    Runs FW 20 times to check variance.
    (This logic is unique, so it doesn't use the standard helper).
    """
    A = secom_data
    k = 30
    steps, samples = 600, 80
    n_runs = 100
    objs = []
    times = []

    print(f"\n=== STABILITY TEST (Runs={n_runs}) ===")
    helper = GreedySolver(A, k)
    solver = FWHomotopySolver(A, k, n_steps=steps, n_mc_samples=samples)

    for i in range(n_runs):
        t0 = time.time()
        s = solver.solve(verbose=False)
        times.append(time.time() - t0)

        indices = np.where(s > 0.5)[0]
        obj = helper.calculate_obj(list(indices))
        objs.append(obj)
        print(f"Run {i+1}: Obj={obj:.4f}")

    mean_obj = np.mean(objs)
    avg_time = np.mean(times)
    cv = (np.std(objs) / mean_obj) * 100

    print(f"Variation: {cv:.2f}% | Avg Time: {avg_time:.4f}s")

    # Logging specialized for stability
    status = "PASS" if cv < 5.0 else "FAIL"
    benchmark_logger(
        "stability_check",
        k,
        steps,
        samples,
        fw_obj=mean_obj,
        fw_time=avg_time,
        status=f"{status} (CV={cv:.1f}%)",
    )

    assert cv < 5.0, f"Unstable! CV: {cv:.2f}%"
