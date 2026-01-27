import numpy as np
import pytest
from grad_fw.benchmarks import run_experiment, find_critical_k
from grad_fw.data_loader import DATASETS


# 5 separate tests, 30 points per iteration
N_TEST = 5
N_POINT = 30

DATASETS = ["residential"]


"""
Experiment 1: The "Cost of Competitiveness" (Effect on kc​)

Hypothesis: Since FW time is linear in n (tFW​∝n) and Greedy time is cubic in k (tGreedy​∝k3), the critical crossover point kc​ should scale roughly as kc​∝n1/3.

    Setup: Fix p (e.g., p=200). Vary steps n (e.g., 100, 200, 400, 800, 1600).

    Measure: The Critical k (kc​) for Time.

    Goal: Plot n vs. kc​. If the curve is flat, our algorithm is extremely robust. If it shoots up, we know that "high precision = low competitiveness."

Experiment 2: The "Convergence Heatmap" (Accuracy vs. n,m)

Hypothesis: Accuracy will improve rapidly and then hit a "plateau" (diminishing returns).

    Setup: Fix p=200 and k=50 (a typical case).

    Vary: Steps n (x-axis) and Samples m (y-axis).

    Measure: Accuracy Ratio (ObjFW​/ObjGreedy​).

    Goal: Find the "Sweet Spot"—the minimum n and m required to beat Greedy (Ratio ≤1.0).
"""


@pytest.mark.parametrize("run_id, num_points", [(i, N_POINT) for i in range(N_TEST)])
@pytest.mark.parametrize("dataset_data", DATASETS, indirect=True)
def test_nmc_fix_p_kc(
    dataset_data, critical_k_logger, critical_k_nmc_logger, run_id, num_points
):
    """Experiment 2: Vary n_mc at fixed p."""
    A_full, name = dataset_data
    p_full = A_full.shape[0]
    nmc_values = np.linspace(25, 2000, 15, dtype=int)

    # p_values = np.linspace(25, p_full, num=num_points, dtype=int)
    # p_values = np.unique(p_values)
    steps = 800
    results = {}
    for nmc in nmc_values:
        # Test Time
        res = find_critical_k(
            A_full,
            name_p=name,
            logger=critical_k_logger,
            max_run=10,
            isTime=True,
            samples=nmc,
            steps=steps,
        )
        best_k = res.get("k", -1)
        results[nmc] = best_k

        # Log final
        critical_k_nmc_logger(
            dataset_name=name,
            p=p_full,
            final_critical_k=best_k,
            speedup=res.get("speedupx"),
            ratio=res.get("ratio"),
            steps=800,
            samples=nmc,
        )

    print(results)
    return
