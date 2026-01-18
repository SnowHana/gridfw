import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver
from grad_fw.data_loader import load_residential_building_data


def test_real_world_benchmark():
    print("\n\n=== REAL-WORLD BENCHMARK: Residential Building Data ===")

    # 1. Load Real Data
    A, _ = load_residential_building_data()
    if A is None:
        pytest.skip("Could not load UCI dataset.")

    p = A.shape[0]  # Should be 103
    k = 10  # Select top 10 representative features

    print(f"Problem: Select k={k} features from p={p}")

    # -------------------------------------------------------
    # COMPETITOR 1: Greedy Algorithm (The Baseline)
    # -------------------------------------------------------
    print("\n[1] Running Greedy Solver...")
    greedy = GreedySolver(A, k)
    g_indices, g_obj, g_time = greedy.solve()

    print(f"    Selected: {np.sort(g_indices)}")
    print(f"    Objective: {g_obj:.4f}")
    print(f"    Time: {g_time:.4f}s")

    # -------------------------------------------------------
    # COMPETITOR 2: FW-Homotopy (Your Algorithm)
    # -------------------------------------------------------
    print("\n[2] Running FW-Homotopy...")
    # Using Unified Robust Settings (Alpha=0.01)
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=1000, n_mc_samples=100)

    start = time.time()
    s_fw = solver.solve(verbose=False)
    fw_time = time.time() - start

    fw_indices = np.where(s_fw > 0.5)[0]

    # Calculate Objective
    # NEW (Correct)
    fw_obj = greedy.calculate_obj(list(fw_indices))

    print(f"    Selected: {np.sort(fw_indices)}")
    print(f"    Objective: {fw_obj:.4f}")
    print(f"    Time: {fw_time:.4f}s")

    # -------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------
    ratio = fw_obj / g_obj
    speedup = g_time / fw_time

    print("\n--- FINAL RESULTS ---")
    print(f"Approximation Ratio (FW / Greedy): {ratio:.2%}")
    print(f"Speedup Factor (Greedy Time / FW Time): {speedup:.2f}x")

    # Criterion: Must be >95% accurate
    assert ratio > 0.95, f"FW performed poorly on real data! Ratio: {ratio:.2f}"

    # Criterion: Should be reasonably fast (not strictly enforcing speedup for small k)
    if speedup > 1.0:
        print("SUCCESS: FW is faster and accurate!")
    else:
        print(
            "NOTE: FW was slower (expected for small k, but should scale better for large k)."
        )


if __name__ == "__main__":
    test_real_world_benchmark()
