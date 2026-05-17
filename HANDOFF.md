# Project Handoff: GridFW + Index Replication Demo

This document is a context handoff for a new chat session working on the
`cssp-index-replication` demo repo that uses `gridfw` as a dependency.

---

## 1. What GridFW is

**GridFW** (`github.com/SnowHana/gridfw`) is a Python library implementing
the Frank-Wolfe Homotopy algorithm for the Column Subset Selection Problem
(CSSP), based on the thesis:

> KIM, Wujin (Daniel). *Scalable Column Subset Selection via Boolean
> Relaxation and Frank-Wolfe Method*. 2026.

### The problem

Given a data matrix X ∈ ℝ^(n×p), find k columns that best reconstruct the
full matrix:

    max   Tr(X^T P_S X)
    s ∈ {0,1}^p, |s| ≤ k

This is NP-hard. Standard approaches: exact (exponential) or greedy (O(pk³)).

### The key idea

We adapt the **Boolean relaxation** framework from minimum-variance portfolio
optimization (Moka et al., 2025) and show it applies to CSSP. The inner
subproblem has the same algebraic form as a portfolio problem.

The relaxed objective g_δ(t) over t ∈ [0,1]^p is:
- Strictly convex when δ ≥ η₁ (largest eigenvalue of A = X^T X / n)
- Agrees with the original at every binary corner point s ∈ {0,1}^p

**FW-Homotopy** runs a geometric schedule δ₀ → η₁:
1. Start in convex regime (unique global min, easy)
2. Decay δ geometrically each step
3. At each step: estimate gradient via Monte Carlo Rademacher sampling,
   run Frank-Wolfe LMO (pick k columns with smallest gradient components),
   update t ← (1-α)t + αs

### Public API

```python
from grad_fw import FWHomotopySolver

solver = FWHomotopySolver(
    A,              # p×p correlation matrix (X^T X / n)
    k=20,           # number of columns to select
    alpha=0.1,      # step size — optimal range [0.1, 0.2]
    n_steps=500,    # default: max(1000, 20k)
    n_mc_samples=50 # Rademacher samples — m ∈ [50, 200] is sufficient
)
s = solver.solve(verbose=False)
selected_indices = np.where(s > 0.5)[0]  # exactly k indices
```

Also available:
```python
from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.benchmarks.BruteForceSolver import BruteForceSolver
from grad_fw.benchmarks.benchmarks import run_experiment
```

`run_experiment(A, k, experiment_name)` runs both FW and Greedy and returns a
dict with keys: `fw_obj`, `g_obj`, `ratio`, `fw_time`, `g_time`, `speedupx`.

### When to use FW-Homotopy vs Greedy (from paper results)

| Condition             | Recommendation                          |
|-----------------------|-----------------------------------------|
| k/p > 0.2, large p   | FW-Homotopy (faster, within 3% quality) |
| k/p < 0.1            | Greedy (FW is unstable in sparse regime)|
| p < 50               | Brute-force or Greedy                   |

Benchmarked on 8 datasets (p=103 to 639). Speedup up to 3.5× for k/p > 0.2.

### Repo structure

```
src/grad_fw/
├── fw_homotomy.py          # main algorithm
├── data_loader.py          # synthetic + real dataset loaders
├── benchmarks/
│   ├── GreedySolver.py
│   ├── BruteForceSolver.py
│   └── benchmarks.py       # run_experiment, find_critical_k
└── verif/
    ├── core.py             # BooleanRelaxation math & gradients
    └── verifiers.py        # gradient checkers (analytical vs numerical)

tests/
├── sanity_check/           # correctness tests
└── grad_check/             # gradient verification
```

---

## 2. What was done to the gridfw repo

Starting from research code, the following was cleaned up:

### Bugs fixed
- `fw_homotomy.py`: Removed dead `best_val`/`best_s` variables that were
  initialized but never used or returned
- `test_algo_opt_1.py`: Fixed gap sign error — was `(opt_obj - fw_obj) /
  |opt_obj|` which is always ≤ 0 for minimization, making the assertion
  trivially pass. Corrected to `(fw_obj - opt_obj) / |opt_obj|`
- `verif/verifiers.py`: Implemented `run_stress_test()` and `report()` on
  both verifier classes — they were empty stubs (`pass`), causing
  `verif/main.py` to crash with AttributeError

### Test cleanup
- Removed dead `restarts` variable in `test_sanity_check.py`
- Fixed `np.sum(s_fw)` → `np.sum(s_fw > 0.5)` for cardinality check
- Removed flaky `assert speedup > 1.0` from `test_secom.py` (hardware-
  dependent, not a correctness property)
- Noted `verify_cssp.py` is broken (calls non-existent `n_restarts` param)
  — should be deleted before release

### Repo cleanup
- Added 249 files to `.gitignore` and untracked them: `logs/`, `data/`,
  `experiment/`, `notebooks/`, `scripts/old/`, `src/gridfw.egg-info/`,
  `__pycache__/`, `.DS_Store`, stray files in `src/`
- Updated `pyproject.toml`: real author name/email, correct GitHub URLs,
  added missing dependencies (`scikit-learn`, `ucimlrepo`)
- Rewrote `README.md` with algorithm explanation, results table, finance
  connection, and "when to use" guidance

### Tests that pass (run with: pytest tests/sanity_check/ tests/grad_check/)
- `test_sanity_check.py` — diagonal sanity, FW vs brute-force, cardinality
- `test_algo_opt_1.py` — stricter brute-force comparison
- `tests/grad_check/test_f_grad.py` — gradient verification for f
- `tests/grad_check/test_g_grad.py` — gradient verification for g (boundary,
  extreme conditioning, algorithm path simulation)

### Tests excluded from release (need local data files, slow)
- `test_residential.py` — requires `data/residential.xlsx`
- `test_secom.py` — requires `data/secom.data`

---

## 3. New repo: cssp-index-replication

### Goal

A standalone demo notebook showing FW-Homotopy applied to S&P 500 index
replication. Target audience: quant interviewers.

The pitch: "Select k stocks whose covariance structure best spans the S&P 500,
using a Frank-Wolfe algorithm that is 3.5× faster than greedy forward
selection for k/p > 0.2."

### Repo structure

```
cssp-index-replication/
├── notebook.ipynb      # single self-contained notebook
├── requirements.txt
└── README.md           # one chart + install + run
```

### requirements.txt

```
git+https://github.com/SnowHana/gridfw.git
yfinance
matplotlib
jupyter
pandas
numpy
```

### Notebook outline (5 sections)

**Section 1 — Motivation**
- Index replication problem: track S&P 500 with k << 500 stocks
- Why covariance structure matters (Markowitz connection)
- Why CSSP is the right formulation

**Section 2 — Data**
```python
import yfinance as yf
import numpy as np

# ~100 large-cap S&P 500 constituents, 2 years training
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", ...]  # 100 tickers
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")["Close"]
returns = data.pct_change().dropna()

# Build correlation matrix (training set)
X_train = returns.to_numpy()
X_train = (X_train - X_train.mean(0)) / X_train.std(0)
A = X_train.T @ X_train / len(X_train)

# Hold-out for evaluation
data_test = yf.download(tickers, start="2024-01-01", end="2025-01-01")["Close"]
returns_test = data_test.pct_change().dropna().to_numpy()
```

**Section 3 — Run CSSP**
```python
from grad_fw import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver

k = 20  # select 20 stocks (k/p = 0.2, right at the sweet spot)

# FW-Homotopy
solver = FWHomotopySolver(A, k=k, alpha=0.1, n_mc_samples=50)
s = solver.solve()
fw_selected = np.where(s > 0.5)[0]

# Greedy baseline
greedy = GreedySolver(A, k)
greedy_selected, _, _ = greedy.solve()

# Random baseline (average over trials)
n_trials = 100
random_selected = [np.random.choice(len(tickers), k, replace=False)
                   for _ in range(n_trials)]
```

**Section 4 — Evaluate (tracking error)**
```python
def tracking_error(selected_idx, returns_test):
    # Equal-weight portfolio of selected stocks
    portfolio = returns_test[:, selected_idx].mean(axis=1)
    # Equal-weight index (all stocks)
    index = returns_test.mean(axis=1)
    return np.std(portfolio - index) * np.sqrt(252)  # annualised

fw_te = tracking_error(fw_selected, returns_test)
greedy_te = tracking_error(greedy_selected, returns_test)
random_te = np.mean([tracking_error(r, returns_test) for r in random_selected])
```

**Section 5 — Results chart**
Plot tracking error vs k (sweep k from 5 to 50) for all three methods.
Expected result: FW ≈ Greedy, both significantly below Random.

### Key talking points for interview
1. *"Why CSSP and not just picking highest-variance stocks?"*
   — Max-variance ignores correlation. CSSP maximizes projection, capturing
   the full covariance structure.

2. *"Why not just use Greedy?"*
   — Greedy is O(pk³). For k=50, p=500, that's 50³×500 = 6.25B operations.
   FW-Homotopy scales as O(n·p³) independent of k, so it's faster when k is
   large relative to p.

3. *"What are the limitations?"*
   — Unstable in sparse regime (k/p < 0.1) due to initialization sensitivity.
   Not the best choice for very small p. See paper for full analysis.

---

## 4. Install gridfw in new repo

```bash
pip install git+https://github.com/SnowHana/gridfw.git
```

Or clone locally and `pip install -e /path/to/gridfw` for development.
