import pytest
import time
import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks import GreedySolver
from grad_fw.data_loader import load_dataset


def test_real_world_benchmark():
    print("\n\n=== BENCHMARK: Residential Building (Accuracy) ===")

    # Load via string name
    A, _ = load_dataset("residential")
    if A is None:
        pytest.skip("Could not load dataset.")

    p = A.shape[0]
    k = 10

    print(f"Problem: Select k={k} from p={p}")

    # 1. Greedy (Baseline)
    print("\n[1] Running Greedy...")
    greedy = GreedySolver(A, k)
    start = time.time()
    _, g_obj, g_time = greedy.solve()
    print(f"    Greedy: Obj={g_obj:.4f} | Time={g_time:.4f}s")

    # 2. FW-Homotopy (Multi-Start)
    print("\n[2] Running FW-Homotopy (Multi-Start)...")
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=1000, n_mc_samples=200)

    n_restarts = 5
    best_fw_obj = -np.inf
    total_time = 0

    for i in range(n_restarts):
        t0 = time.time()
        s_fw = solver.solve(verbose=False)
        total_time += time.time() - t0

        indices = np.where(s_fw > 0.5)[0]
        obj = greedy.calculate_obj(list(indices))
        if obj > best_fw_obj:
            best_fw_obj = obj

    print(f"    FW Best: Obj={best_fw_obj:.4f} | Avg Time={total_time/n_restarts:.4f}s")

    ratio = best_fw_obj / g_obj
    print(f"--- Result: Ratio {ratio:.2%} ---")

    assert ratio > 0.95, f"Accuracy too low: {ratio:.2f}"


def test_scalability_k40():
    print("\n\n=== BENCHMARK: Residential Building (Scalability k=40) ===")
    A, _ = load_dataset("residential")
    if A is None:
        pytest.skip("No Data")

    p = A.shape[0]
    k = 40

    # 1. Random Baseline (Check if problem is hard)
    print("[1] Random Baseline...")
    rand_objs = []
    helper = GreedySolver(A, k)
    for _ in range(50):
        idx = np.random.choice(p, k, replace=False)
        rand_objs.append(helper.calculate_obj(idx))
    avg_rand = np.mean(rand_objs)
    print(f"    Random Avg: {avg_rand:.4f}")

    # 2. Greedy
    print("\n[2] Greedy...")
    greedy = GreedySolver(A, k)
    t0 = time.time()
    _, g_obj, _ = greedy.solve()
    g_time = time.time() - t0
    print(f"    Greedy: {g_time:.4f}s")

    # 3. FW-Homotopy
    print("\n[3] FW-Homotopy...")
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=600, n_mc_samples=50)
    t0 = time.time()
    s_fw = solver.solve(verbose=False)
    fw_time = time.time() - t0

    indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(indices))
    print(f"    FW:     {fw_time:.4f}s")

    print(f"\n--- Scalability Verdict ---")
    print(f"Improvement over Random: {((fw_obj-avg_rand)/avg_rand)*100:.1f}%")
    print(f"Time Diff: {fw_time - g_time:.4f}s")

    assert fw_obj / g_obj > 0.90, "Optimization collapsed at high k"
