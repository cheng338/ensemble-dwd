"""Chang Cheng's ensemble kernel Distance Weighted Discrimination."""

from .estimator import EnsembleKernelDWD, IneligibleSelectionError
from .model_selection import mnist_param_grid
from .multiclass import OvOClassifier

__version__ = "0.2.1"

__all__ = ["EnsembleKernelDWD", "IneligibleSelectionError", "OvOClassifier", "mnist_param_grid"]
