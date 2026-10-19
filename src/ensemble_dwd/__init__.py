"""Chang Cheng's ensemble kernel DWD for binary and multiclass classification."""

from .estimator import EnsembleKernelDWD, IneligibleSelectionError, IneligibleSubsetError
from .model_selection import mnist_param_grid, mnist_svm_param_grid
from .multiclass import OvOClassifier
from .svm import EnsembleKernelSVM

__version__ = "0.3.13"

__all__ = ["EnsembleKernelDWD", "EnsembleKernelSVM", "IneligibleSelectionError", "IneligibleSubsetError",
           "OvOClassifier", "mnist_param_grid", "mnist_svm_param_grid"]
