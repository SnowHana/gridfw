# GridFW: Gradient Framework for CSSP

**GridFW** is a Python library for solving the **Column Subset Selection Problem (CSSP)** using a **Frank-Wolfe Homotopy** approach. It provides efficient solvers for large-scale subset selection tasks, particularly useful in machine learning and data summarization.

## Features

*   **Frank-Wolfe Homotopy Solver**: A gradient-based method for selecting optimal column subsets.
*   **Greedy Solver**: A baseline implementation for performance comparison.
*   **Modular Design**: Easily extensible for different objective functions.

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
from grad_fw.fw_homotomy import FWHomotopySolver

# Generate synthetic data (A = X^T X)
p = 20
k = 5
X = np.random.randn(100, p)
A = X.T @ X

# Initialize Solver
solver = FWHomotopySolver(A, k, alpha=0.01, n_steps=500)

# Solve
solution = solver.solve()
selected_indices = np.where(solution > 0.5)[0]

print(f"Selected Indices: {selected_indices}")
```

## Project Structure

*   `src/`: Source code for the `grad_fw` package.
*   `experiment/`: Benchmarking scripts.
*   `scripts/`: Utility and verification scripts.
*   `notebooks/`: Jupyter notebooks for data exploration.
*   `tests/`: Unit tests.
*   `cpp/`: (Planned) C++ implementation for performance comparison.

## License

MIT License
