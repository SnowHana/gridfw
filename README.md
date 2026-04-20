# GridFW: Gradient Framework for CSSP

**GridFW** is a Python library for solving the **Column Subset Selection Problem (CSSP)** using a **Frank-Wolfe Homotopy** approach. It provides efficient solvers for large-scale subset selection tasks, particularly useful in machine learning, data summarization, and **quantitative finance** (e.g. portfolio construction and stock factor selection).

## Features

*   **Frank-Wolfe Homotopy Solver**: A gradient-based method for selecting optimal column subsets with provable convergence.
*   **Greedy Solver**: A fast forward-selection baseline for performance comparison.
*   **Brute-Force Solver**: Exact exponential-time solver for small-scale ground-truth validation.
*   **Modular Design**: Easily extensible for new objective functions (e.g. A-Optimality, minimum-variance portfolio).

## Installation

To install the package in editable mode (recommended for development):

```bash
git clone https://github.com/yourusername/gridfw.git
cd gridfw
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from grad_fw import FWHomotopySolver

# Generate synthetic data (A = X^T X / N)
p = 20
k = 5
X = np.random.randn(100, p)
A = X.T @ X / 100

# Initialize and solve
solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=500)
solution = solver.solve()
selected_indices = np.where(solution > 0.5)[0]

print(f"Selected Indices: {selected_indices}")
```

### Loading Real Datasets

```python
from grad_fw import DatasetLoader

loader = DatasetLoader()
A, X_norm = loader.load("synthetic_high_corr", N=2000, p=500)
```

## Project Structure

```
src/grad_fw/
├── __init__.py          # Public API: FWHomotopySolver, DatasetLoader
├── fw_homotomy.py       # Frank-Wolfe Homotopy solver
├── data_loader.py       # Dataset loading & preprocessing
├── benchmarks/
│   ├── GreedySolver.py  # Greedy forward-selection baseline
│   ├── BruteForceSolver.py  # Exact brute-force (small p only)
│   └── benchmarks.py    # run_experiment / find_critical_k utilities
└── verif/
    ├── core.py          # BooleanRelaxation math & gradient formulas
    └── verifiers.py     # Numerical gradient checkers

experiment/              # Benchmarking scripts
scripts/                 # Utility & verification scripts
tests/                   # Pytest suite (grad checks, performance)
notebooks/               # Jupyter notebooks for exploration
```

## License

MIT License
