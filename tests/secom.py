import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver
from grad_fw.data_loader import load_dataset


def test_secom_benchmark():
    print("\n\n=== BENCHMARK: SECOM (Semiconductor) ===")

    # Load via string name
    A, _ = load_dataset("secom")
    if A is None:
        pytest.skip("Could not load SECOM.")

    p = A.shape[0]
    k = 50  # Large k

    print(f"Problem: Select k={k} from p={p}")

    # 1. Greedy
    print("\n[1] Running Greedy...")
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0
    print(f"    Greedy: Obj={g_obj:.4f} | Time={g_time:.4f}s")

    # 2. FW-Homotopy
    print("\n[2] Running FW-Homotopy...")
    # SECOM is noisy and large -> stochastic mode is essential
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=800, n_mc_samples=50)

    t0 = time.time()
    s_fw = solver.solve(verbose=False)
    fw_time = time.time() - t0

    indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(indices))
    print(f"    FW:     Obj={fw_obj:.4f} | Time={fw_time:.4f}s")

    # Verdict
    ratio = fw_obj / g_obj
    speedup = g_time / fw_time

    print("\n--- Result ---")
    print(f"Ratio:   {ratio:.2%}")
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print("VICTORY: FW is faster than Greedy!")

    assert ratio > 0.90, "Performance dropped on SECOM"
