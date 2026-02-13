import numpy as np
import time
from grad_fw.fw_homotomy import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver


def verify():
    print("=== CSSP Maximization Verification ===")

    # 1. Generate Synthetic Problem
    p = 20
    k = 5
    np.random.seed(42)
    X = np.random.randn(100, p)
    A = X.T @ X

    print(f"Problem: p={p}, k={k}")

    # 2. Greedy Solver (Maximization)
    print("\n[1] Running Greedy Solver...")
    greedy = GreedySolver(A, k)
    t0 = time.time()
    g_indices, g_obj, g_time = greedy.solve()
    print(f"    Greedy Indices: {g_indices}")
    print(f"    Greedy Objective: {g_obj:.4f}")
    print(f"    Greedy Time: {g_time:.4f}s")

    # 3. FW-Homotopy Solver (Maximization)
    print("\n[2] Running FW-Homotopy Solver...")
    # Using default objective_type='cssp'
    solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=500, n_mc_samples=50)
    t0 = time.time()
    s_fw = solver.solve(n_restarts=3, verbose=True)
    fw_time = time.time() - t0

    fw_indices = np.where(s_fw > 0.5)[0]
    fw_obj = greedy.calculate_obj(list(fw_indices))

    print(f"\n[3] Results")
    print(f"    FW Indices: {fw_indices}")
    print(f"    FW Objective: {fw_obj:.4f}")
    print(f"    FW Time: {fw_time:.4f}s")

    ratio = fw_obj / g_obj
    print(f"    Ratio (FW/Greedy): {ratio:.4f}")

    if ratio > 0.9:
        print("\nSUCCESS: FW is close to or better than Greedy!")
    else:
        print("\nWARNING: FW performance is low.")


if __name__ == "__main__":
    verify()
