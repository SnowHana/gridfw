import numpy as np
from grad_fw.fw_homotomy import FWHomotopySolver

A = np.random.rand(2, 2)
fw_solver = FWHomotopySolver(A, k=1)
sol = fw_solver.solve_with_history()
