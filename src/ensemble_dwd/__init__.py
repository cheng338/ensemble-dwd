"""Chang Cheng's ensemble kernel DWD and SVM comparison classifiers."""

from .estimator import EnsembleKernelDWD, IneligibleSelectionError
from .model_selection import mnist_param_grid, mnist_svm_param_grid
from .multiclass import OvOClassifier
from .svm import EnsembleKernelSVM

__version__ = "0.3.0"

__all__ = ["EnsembleKernelDWD", "EnsembleKernelSVM", "IneligibleSelectionError",
           "OvOClassifier", "mnist_param_grid", "mnist_svm_param_grid"]
