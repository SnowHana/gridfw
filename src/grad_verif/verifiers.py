import numpy as np
from grad_verif.core import ProblemGenerator, BooleanRelaxation


class BaseVerifier:
    """
    Helper class for Gradient Verification.
    Provides tools to generate problems and check errors
    """

    def __init__(self, h=1e-7, threshold=1e-5):
        """
        Args:
            h (float): Step size for numerical differentiation.
            threshold (float): Max allowed relative error.
        """
        self.h = h
        self.threshold = threshold
        self.gen = ProblemGenerator()
        self.math = BooleanRelaxation()

    def check_error(self, g_ana, g_num):
        """
        Calculates relative error between Analytical (g_ana) and Numerical (g_num) gradients.
        Includes safety logic for tiny signals (The 'Relative Error Trap').
        """
        abs_err = np.linalg.norm(g_ana - g_num)
        norm = np.linalg.norm(g_ana)

        # Avoid division by zero
        rel_err = abs_err / (norm + 1e-10)

        # Trust analytical if both signal and error are negligible (e.g., < 1e-8)
        # Avoid small size error...
        if norm < 1e-8 and abs_err < 1e-8:
            rel_err = 0.0

        return rel_err


class SingleGradientVerifier(BaseVerifier):
    """Specialized for verifying the deterministic function f(t)."""

    pass


class ExpectedGradientVerifier(BaseVerifier):
    """Specialized for verifying the stochastic expectation g(t)."""

    pass
