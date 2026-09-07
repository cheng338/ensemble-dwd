"""Chang Cheng's ensemble kernel Distance Weighted Discrimination."""

from .estimator import EnsembleKernelDWD, IneligibleSelectionError
from .model_selection import mnist_param_grid

__version__ = "0.1.1"

__all__ = ["EnsembleKernelDWD", "IneligibleSelectionError", "mnist_param_grid"]
