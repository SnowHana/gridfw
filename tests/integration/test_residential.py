import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.data_loader import load_dataset_online


def test_medium_accuracy_stability(benchmark_logger):
    """
    Medium-level test on Residential Building dataset (p=103).
    Goals:
    1. Accuracy: FW Objective should be within 1.35x of Greedy (Ratio < 1.35).
       (User noted 1.1-1.3 is promising).
    2. Stability: Coefficient of Variation (CV) < 5%.
    """
    print("\n\n=== BENCHMARK: Residential Building (Medium Setup) ===")

    # Load via string name
    A, _ = load_dataset_online("residential")
    if A is None:
        pytest.skip("Could not load dataset.")

    p = A.shape[0]
    k = 20  # Medium k for p=103
    steps = 2000
    samples = 500

    print(f"Problem: Select k={k} from p={p}")

    # 1. Greedy (Baseline)
    print("\n[1] Running Greedy...")
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, g_time = greedy.solve()
    print(f"    Greedy: Obj={g_obj:.4f} | Time={g_time:.4f}s")

    # 2. FW-Homotopy (Stability Run)
    n_runs = 10
    print(f"\n[2] Running FW-Homotopy ({n_runs} runs for stability)...")

    # Using parameters that balance speed and accuracy for medium setup
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=steps, n_mc_samples=samples)

    fw_objs = []
    total_time = 0

    for i in range(n_runs):
        t0 = time.time()
        # Single restart per run to test stability of the stochastic algorithm itself
        s_fw = solver.solve(n_restarts=1, verbose=False)
        run_time = time.time() - t0
        total_time += run_time

        indices = np.where(s_fw > 0.5)[0]
        obj = greedy.calculate_obj(list(indices))
        fw_objs.append(obj)
        # print(f"    Run {i+1}: {obj:.4f}")

    fw_mean = np.mean(fw_objs)
    fw_std = np.std(fw_objs)
    cv = fw_std / fw_mean if fw_mean != 0 else 0.0

    ratio = fw_mean / g_obj

    print(f"\n--- Results ---")
    print(f"    FW Mean Obj: {fw_mean:.4f}")
    print(f"    FW Std Dev:  {fw_std:.4f}")
    print(f"    FW CV:       {cv:.2%}")
    print(f"    Ratio (FW/G): {ratio:.4f}")
    print(f"    Avg Time:    {total_time/n_runs:.4f}s")

    # Log Result
    status = "PASS" if (ratio > 0.75 and cv < 0.05) else "FAIL"
    benchmark_logger(
        "residential_medium_stability",
        k,
        steps,
        samples,
        g_obj=g_obj,
        fw_obj=fw_mean,
        g_time=g_time,
        fw_time=total_time / n_runs,
        status=f"{status} (CV={cv:.1f}%)",
        dataset_name="Residential",
    )

    # Assertions based on user goals (Maximization)
    # Ratio > 0.75 means FW is at least 75% of Greedy
    assert ratio > 0.75, f"Accuracy too low: Ratio {ratio:.2f} < 0.75"

    # CV should be around 3% (assert < 5% for safety)
    assert cv < 0.05, f"Instability detected: CV {cv:.2%} > 5%"


def test_scalability_k40(benchmark_logger):
    print("\n\n=== BENCHMARK: Residential Building (Scalability k=40) ===")
    A, _ = load_dataset_online("residential")
    if A is None:
        pytest.skip("No Data")

    p = A.shape[0]
    k = 40
    steps = 600
    samples = 50

    # 1. Greedy
    print("\n[1] Greedy...")
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0
    print(f"    Greedy: Obj={g_obj:.4f} ({g_time:.4f}s)")

    # 2. FW-Homotopy
    print("\n[2] FW-Homotopy...")
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=steps, n_mc_samples=samples)
    t0 = time.time()
    s_fw = solver.solve(n_restarts=1, verbose=False)
    fw_time = time.time() - t0

    indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(indices))

    ratio = fw_obj / g_obj
    print(f"    FW: Obj={fw_obj:.4f} ({fw_time:.4f}s)")
    print(f"    Ratio: {ratio:.4f}")

    status = "PASS" if ratio > 0.70 else "FAIL"
    benchmark_logger(
        "residential_scalability_k40",
        k,
        steps,
        samples,
        g_obj=g_obj,
        fw_obj=fw_obj,
        g_time=g_time,
        fw_time=fw_time,
        status=status,
        dataset_name="Residential",
    )

    # Allow slightly lower ratio for larger k/scalability test
    assert ratio > 0.70, f"Optimization collapsed at high k: Ratio {ratio:.2f}"
