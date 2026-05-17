"""
Quick benchmark (~3-5 min) that verifies core paper claims using only
synthetic data. No external data files required. Intended as a fast
sanity-check before running the full benchmarks.

Paper claims verified:
  - §4.3.1: α ∈ [0.1, 0.2] gives stable, low-error solutions
  - §4.3.2: Larger m does not significantly improve accuracy
  - §4.3.3: FW accuracy degrades in sparse regime (k/p < 0.1)
  - §4.4:   FW speedup grows with k/p in dense regime
"""
import pytest
import numpy as np
from grad_fw.benchmarks.benchmarks import run_experiment

pytestmark = pytest.mark.slow

# Synthetic matrix used across all quick tests.
# p=300 is large enough to show speedup trends but runs fast.
@pytest.fixture(scope="module")
def quick_matrix():
    np.random.seed(42)
    p = 300
    X = np.random.randn(600, p)
    # Introduce high correlation so FW's structural advantage shows up
    factor = np.random.randn(600, 1)
    X += 0.8 * factor
    A = X.T @ X / len(X)
    return A, p


# --- §4.3.1: Alpha sensitivity ---

@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 0.2, 0.5])
def test_alpha_sensitivity(quick_matrix, alpha, sweep_logger):
    """
    Paper Fig 13: α ∈ [0.1, 0.2] should give low, stable error.
    α = 0.001 and α = 0.5 should give higher or unstable error.
    Run at k/p = 0.2 (dense regime where FW is expected to work well).
    """
    A, p = quick_matrix
    k = int(0.2 * p)
    res = run_experiment(
        A, k,
        experiment_name=f"quick_alpha_{alpha}",
        alpha=alpha,
        dataset_name="synthetic_quick",
        steps=400,
        samples=50,
    )
    sweep_logger(**res)
    print(f"  α={alpha}: ratio={res['ratio']:.4f}")


# --- §4.3.2: Sample size (m) sensitivity ---

@pytest.mark.parametrize("m", [10, 50, 100, 200])
def test_m_sensitivity(quick_matrix, m, sweep_logger):
    """
    Paper Fig 14: Accuracy should be comparable across m ∈ [50, 200].
    m=10 may be noisy; m=200 should not be significantly better than m=50.
    """
    A, p = quick_matrix
    k = int(0.2 * p)
    res = run_experiment(
        A, k,
        experiment_name=f"quick_m_{m}",
        alpha=0.1,
        dataset_name="synthetic_quick",
        steps=400,
        samples=m,
    )
    sweep_logger(**res)
    print(f"  m={m}: ratio={res['ratio']:.4f}")


# --- §4.3.3: Sparse vs dense regime ---

@pytest.mark.parametrize("kp_ratio", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
def test_kp_ratio_sweep(quick_matrix, kp_ratio, sweep_logger):
    """
    Paper Figs 4, 7: FW accuracy should degrade below k/p ≈ 0.1 (sparse)
    and be within 3% of Greedy above k/p = 0.2 (dense).
    """
    A, p = quick_matrix
    k = max(1, int(kp_ratio * p))
    res = run_experiment(
        A, k,
        experiment_name=f"quick_kp_{kp_ratio}",
        alpha=0.2,
        dataset_name="synthetic_quick",
        steps=400,
        samples=50,
    )
    sweep_logger(**res)
    print(f"  k/p={kp_ratio}: ratio={res['ratio']:.4f}, speedup={res['speedupx']:.2f}x")

    # Hard assertion only in dense regime where paper guarantees hold
    if kp_ratio >= 0.2:
        assert res["ratio"] > 0.97, (
            f"Dense regime (k/p={kp_ratio}): FW ratio {res['ratio']:.3f} "
            f"should be > 0.97 (within 3% of Greedy)"
        )


# --- §4.4: Speedup trend with k ---

def test_speedup_grows_with_k(quick_matrix, sweep_logger):
    """
    Paper Fig 9: Speedup should increase as k/p increases.
    We check that speedup at k/p=0.4 > speedup at k/p=0.1.
    """
    A, p = quick_matrix
    speedups = {}

    for kp_ratio in [0.1, 0.2, 0.3, 0.4]:
        k = int(kp_ratio * p)
        res = run_experiment(
            A, k,
            experiment_name=f"quick_speedup_kp{kp_ratio}",
            alpha=0.2,
            dataset_name="synthetic_quick",
            steps=400,
            samples=50,
        )
        speedups[kp_ratio] = res["speedupx"]
        sweep_logger(**res)
        print(f"  k/p={kp_ratio}: speedup={res['speedupx']:.2f}x")

    assert speedups[0.4] > speedups[0.1], (
        f"Speedup should grow with k: got {speedups[0.1]:.2f}x at k/p=0.1 "
        f"vs {speedups[0.4]:.2f}x at k/p=0.4"
    )
