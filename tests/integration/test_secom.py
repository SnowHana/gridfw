import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver
from grad_fw.data_loader import load_dataset


@pytest.fixture(scope="module")
def secom_data():
    print("\n[Setup] Loading SECOM Data...")
    A, _ = load_dataset("secom")
    if A is None:
        pytest.skip("Could not load SECOM data")
    return A


def run_comparison(test_name, A, k, steps, samples, logger, threshold=0.80, restarts=1):
    print(f"\n=== {test_name} (k={k}) ===")

    # 1. Greedy (Baseline)
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0

    # 2. FW-Homotopy (Challenger)
    solver = FWHomotopySolver(A, k, n_steps=steps, n_mc_samples=samples)
    t0 = time.time()
    s_fw = solver.solve(n_restarts=restarts, verbose=False)  # Now safe to call
    fw_time = time.time() - t0

    # 3. Metrics
    fw_indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(fw_indices))

    # Ratio: Higher is Better (Maximization)
    ratio = fw_obj / g_obj if g_obj != 0 and not np.isinf(g_obj) else 0.0
    speedup = g_time / fw_time if fw_time > 0 else 0.0

    print(f"Greedy: {g_time:.4f}s | Obj: {g_obj:.4f}")
    print(f"FW:     {fw_time:.4f}s | Obj: {fw_obj:.4f}")
    print(f"Stats:  Ratio={ratio:.2f} | Speedup={speedup:.2f}x")

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
        dataset_name="SECOM",
    )
    return ratio, speedup


# --- TESTS ---


def test_secom_accuracy_k10(secom_data, benchmark_logger):
    """Small k=10. Focus: High Accuracy."""
    ratio, _ = run_comparison(
        "accuracy_k10",
        secom_data,
        k=10,
        steps=500,
        samples=50,
        logger=benchmark_logger,
        threshold=0.90,
    )
    assert ratio > 0.90, f"Accuracy poor. Ratio: {ratio:.2f}"


def test_secom_benchmark_k50(secom_data, benchmark_logger):
    """Medium k=50. Focus: Speedup & Accuracy."""
    ratio, speedup = run_comparison(
        "benchmark_k50",
        secom_data,
        k=50,
        steps=800,
        samples=30,
        logger=benchmark_logger,
        threshold=0.85,
    )
    assert ratio > 0.85, "Accuracy dropped."
    assert speedup > 1.0, "FW should be faster than Greedy"


def test_secom_stress_k100(secom_data, benchmark_logger):
    """Large k=100. Using Restarts."""
    ratio, speedup = run_comparison(
        "stress_k100",
        secom_data,
        k=100,
        steps=400,
        samples=30,
        restarts=3,
        logger=benchmark_logger,
        threshold=0.80,
    )
    assert ratio > 0.80, f"Accuracy too low. Ratio: {ratio:.2f}"
    assert speedup > 1.0, f"Speedup insufficient! Got {speedup:.2f}x"


def test_secom_stability(secom_data, benchmark_logger):
    """Variance Check."""
    A = secom_data
    k = 30
    steps, samples = 500, 30
    n_runs = 10
    objs = []

    helper = GreedySolver(A, k)
    solver = FWHomotopySolver(A, k, n_steps=steps, n_mc_samples=samples)

    print(f"\n=== STABILITY TEST (Runs={n_runs}) ===")
    for i in range(n_runs):
        s = solver.solve(n_restarts=1, verbose=False)
        idx = np.where(s > 0.5)[0]
        obj = helper.calculate_obj(list(idx))
        objs.append(obj)

    mean_obj = np.mean(objs)
    cv = (np.std(objs) / mean_obj) * 100
    print(f"Variation: {cv:.2f}%")

    status = "PASS" if cv < 5.0 else "FAIL"
    benchmark_logger(
        "stability_check",
        k,
        steps,
        samples,
        fw_obj=mean_obj,
        status=f"{status} (CV={cv:.1f}%)",
        dataset_name="SECOM",
    )
    assert cv < 5.0, f"Unstable! CV: {cv:.2f}%"
