# GridFW: Scalable Column Subset Selection via Boolean Relaxation and Frank-Wolfe

Python implementation of [_Scalable Column Subset Selection via Boolean Relaxation and Frank-Wolfe Method_](CSSP.pdf) (KIM, 2026).

## What it does

The **Column Subset Selection Problem (CSSP)** asks: given a data matrix $X \in \mathbb{R}^{(n×p)}$, find the $k$ columns that best reconstruct the full matrix under the projection objective. Formally:

$$
\max   \text{Tr}(X^T P_S X) \quad
s ∈ \{0,1\}^p, |s| \leq k.
$$

This is NP-hard.
The standard approaches are either exact but exponential (branch-and-bound) or fast but greedy (forward selection).
This library takes a third path.

## Key idea

We adapt the **Boolean relaxation** framework of Moka et al. (2025) — originally designed for minimum-variance portfolio selection — and show it applies directly to CSSP.
The key observation is that the CSSP inner subproblem has the same algebraic form as a minimum-variance portfolio problem, which gives us a continuous relaxation with provable properties.

The relaxed objective $g_\delta(t)$ over $t \in [0,1]^p$ is:

- **Strictly convex** when $\delta \geq \eta_1$ (largest eigenvalue of
  $A = X^T X / n$)
- **Agrees with the original** at every binary corner point $s \in \{0,1\}^p$

**FW-Homotopy** exploits this by running a geometric schedule
$\delta_0 \to \eta_1$.
It starts in the convex regime (unique global minimum, easy to find), then gradually transitions to the harder non-convex regime while tracking the solution.
At each step, the gradient is estimated via Monte Carlo Rademacher sampling and the Frank-Wolfe LMO selects the $k$ columns with the smallest gradient components.

This directly mirrors index replication in quantitative finance: selecting k assets whose covariance structure best spans the full index.

### Demo

![2 Variable 3d Plot](fw_landscape_2d.gif)

Observe how as $\delta$ increases, surface changes toward a concavity, and objective value converges to the solution (Corner point).

## Results

Benchmarked on 8 datasets ($p = 103$ to $639$) against Greedy forward selection:

| Regime               | Accuracy vs Greedy | Speedup           |
| -------------------- | ------------------ | ----------------- |
| Dense ($k/p > 0.2$)  | Within 3%          | Up to 3.5× faster |
| Sparse ($k/p < 0.1$) | Higher variance    | Greedy preferred  |

**Recommended parameters:** $\alpha \in [0.1, 0.2], m \in [50, 200]$. Larger $m$ does not significantly improve objective quality — $n$ (steps) and $\alpha$ are the dominant factors.

## Installation

```bash
pip install git+https://github.com/SnowHana/gridfw.git
```

Or in editable mode for development:

```bash
git clone https://github.com/SnowHana/gridfw.git
cd gridfw
pip install -e .
```

## Usage

```python
import numpy as np
from grad_fw import FWHomotopySolver

# Build correlation matrix from data (e.g. asset returns)
X = np.random.randn(252, 100)   # 252 trading days, 100 assets
A = X.T @ X / len(X)

# Select k=20 assets that best span the covariance structure
solver = FWHomotopySolver(A, k=20, alpha=0.1, n_steps=500, n_mc_samples=50)
s = solver.solve()
selected = np.where(s > 0.5)[0]

print(f"Selected assets: {selected}")  # exactly 20 indices
```

For comparison against the Greedy baseline:

```python
from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.benchmarks.benchmarks import run_experiment

result = run_experiment(A, k=20, experiment_name="my_experiment")
print(f"FW/Greedy ratio: {result['ratio']:.3f} | Speedup: {result['speedupx']:.2f}x")
```

## Project Structure

```
src/grad_fw/
├── __init__.py          # Public API: FWHomotopySolver, DatasetLoader
├── fw_homotomy.py       # FW-Homotopy solver (main algorithm)
├── data_loader.py       # Dataset loading & preprocessing
├── benchmarks/
│   ├── GreedySolver.py      # Greedy forward-selection baseline  O(pk^3)
│   ├── BruteForceSolver.py  # Exact brute-force (small p only)
│   └── benchmarks.py        # run_experiment / find_critical_k
└── verif/
    ├── core.py          # BooleanRelaxation math & gradient formulas
    └── verifiers.py     # Numerical gradient checkers

tests/
├── sanity_check/        # Correctness tests (diagonal, brute-force comparison)
├── grad_check/          # Gradient verification (analytical vs numerical)
└── performance/         # Paper replication experiments (marked slow)
```

## Reproducing paper results

**Correctness tests** (fast, no external data needed):

```bash
pytest tests/sanity_check/ tests/grad_check/
```

**Numerical experiments** from the paper (slow — sweep over k, steps, n_mc, α, and p):

```bash
pytest tests/performance/ -m slow
```

Results are logged as CSV files to `logs/` (created automatically).

### Dataset availability

| Dataset                       | Source                                                                                      | Required action                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Synthetic, Synthetic Toeplitz | Generated in code                                                                           | None                                                                     |
| MNIST, Madelon                | OpenML (auto-download)                                                                      | None                                                                     |
| Myocardial                    | UCI Repo (auto-download)                                                                    | None                                                                     |
| SECOM                         | [UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom)                          | Download `secom.data` → `data/secom.data`                                |
| Residential Building          | [UCI ML Repository](https://archive.ics.uci.edu/dataset/437/residential+building+data+sets) | Download `Residential-Building-Data-Sets.xlsx` → `data/residential.xlsx` |
| Arrhythmia                    | [UCI ML Repository](https://archive.ics.uci.edu/dataset/5/arrhythmia)                       | Download `arrhythmia.data` → `data/arrhythmia.data`                      |

Tests that cannot find their data file are automatically skipped.

## When to use FW-Homotopy vs Greedy

| Condition                      | Recommendation                           |
| ------------------------------ | ---------------------------------------- |
| $k/p > 0.2$ and $p$ is large   | FW-Homotopy (faster, comparable quality) |
| $k/p < 0.1$ (sparse selection) | Greedy (more stable)                     |
| $p$ is small ($< 50$)          | Brute-force or Greedy                    |

## Reference

KIM, Wujin (Daniel). _Scalable Column Subset Selection via Boolean Relaxation and Frank-Wolfe Method_. 2026.

Based on the Boolean relaxation framework of Moka et al. (2025), originally developed for minimum-variance portfolio optimization.

## License

MIT License
