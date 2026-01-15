import numpy as np


def get_convexity_delta(A, k, p, n):
    """_summary_

    Args:
        A (_type_): Matrix X^T X
        k (_type_): Max num of selection
        p (_type_): A's dimension (A = p x p)
        n (_type_): Num iteration of delta
    """
    # Get delta value where function changes its convexity
    # eta1
    eigenvals = np.linalg.eigvalsh(A)
    eta_p, eta_1 = eigenvals[0], eigenvals[-1]
    eps = 0.1 * (k / p)
    delta_0 = 3 * eta_p * (eps**2) / (1 + 3 * (eps**2))
    r = (eta_1 / delta_0) ** (1 / (n - 1))


def verify_gradient_with_stats():
    p = int(input("Enter dimension (p): "))
    num_matrices = int(input("Number of matrices (n): "))
    num_points = int(input("Number of points per matrix (m): "))

    all_errors = []

    for i in range(num_matrices):
        X = np.random.randn(p, p)
        A = X.T @ X  # Symmetric positive definite [cite: 55, 84]

        # Test a delta from the algorithm range [cite: 205]
        delta = np.random.uniform(0.1, 10.0)

        # # Calculate Eigenvalues for the Delta ranges
        # eigenvals = np.linalg.eigvalsh(A)
        # eta_p, eta_1 = eigenvals[0], eigenvals[-1]

        # # Algorithmic delta_0 based on eta_p
        # eps = 0.1 * (1.0 / p)
        # delta_0 = (3 * eta_p * eps**2) / (1 + 3 * eps**2)

        for j in range(num_points):
            t = np.random.uniform(0.1, 0.9, size=p)
            xi = np.random.choice([-1, 1], size=p)  # Rademacher [cite: 65]
            b = A @ xi  # b^(xi) definition [cite: 56]

            # Use your existing single_point function logic
            error = single_point_verify_gradient_logic(p, delta, t, b, A)
            all_errors.append(error)

    # Statistical Summary
    print("\n--- Verification Statistics ---")
    print(f"Mean Error:   {np.mean(all_errors):.2e}")
    print(f"Median Error: {np.median(all_errors):.2e}")
    print(f"Max Error:    {np.max(all_errors):.2e}")
    print(f"Min Error:    {np.min(all_errors):.2e}")

    threshold = 1e-7
    if np.max(all_errors) < threshold:
        print(f"SUCCESS: All errors are below threshold ({threshold})")
    else:
        print(f"WARNING: Some errors exceed threshold.")


def single_point_verify_gradient_logic(p, delta, t, b, A, h=1e-7):
    # This is essentially your code's core logic
    def calculate_f(t_vec):
        Dt = np.diag(1.0 / (t_vec**2)) - np.eye(p)
        Pi_t = A + delta * Dt
        return b.T @ np.linalg.inv(Pi_t) @ b

    # Analytical [cite: 148, 225]
    Pi_inv = np.linalg.inv(A + delta * (np.diag(1.0 / (t**2)) - np.eye(p)))
    grad_a = 2 * delta * ((Pi_inv @ b) ** 2) / (t**3)

    # Numerical
    grad_n = np.zeros(p)
    for i in range(p):
        t_plus = t.copy()
        t_plus[i] += h
        t_minus = t.copy()
        t_minus[i] -= h
        grad_n[i] = (calculate_f(t_plus) - calculate_f(t_minus)) / (2 * h)

    return np.linalg.norm(grad_a - grad_n)


verify_gradient_with_stats()
