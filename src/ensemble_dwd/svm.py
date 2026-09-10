"""Binary ensemble SVM with signed-margin selection and a final SVC refit."""

from numbers import Integral
from os import PathLike
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cache import _cached_base_stage, _runtime_provenance
from .estimator import (
    IneligibleSelectionError,
    _finite_real,
    _fraction,
    _overflow_safe_consensus,
    _positive_integer,
    _private_random_state,
    _sample_subsets,
)


def _batched_scores(model, X, batch_size):
    """Read one finite binary SVC decision value per feature row."""
    size = len(X) if batch_size is None else _positive_integer(
        batch_size, "prediction_batch_size")
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), size):
        stop = min(start + size, len(X))
        values = np.asarray(model.decision_function(X[start:stop]), dtype=np.float64)
        if values.shape != (stop - start,) or not np.isfinite(values).all():
            raise FloatingPointError("SVC must return one finite decision value per observation.")
        scores[start:stop] = values
    return scores


def _diagnostics(model):
    """Normalize the diagnostics of one fitted binary SVC."""
    return dict(n_iter=int(model.n_iter_[0]), fit_status=int(model.fit_status_),
                n_support=np.asarray(model.n_support_, dtype=int).copy())


class EnsembleKernelSVM(ClassifierMixin, BaseEstimator):
    """Fit sampled SVCs, select near signed consensus margins, and refit.

    Each of ``M`` subsets contains ``max(2, floor(theta * n))`` rows sampled
    uniformly without replacement, conditional on including both classes.
    Subsets may overlap. The base decision values on all supplied training
    rows are averaged without normalization. With labels encoded as -1 for
    ``classes_[0]`` and +1 for ``classes_[1]``, rows are ranked by
    ``abs(y_signed * consensus - 1)``. A fresh SVC is fitted on the first
    ``floor(sv_selection_ratio * n)`` ranked rows. Ties retain input order.

    The signed criterion differs from the unsigned ``abs(abs(consensus)-1)``
    criterion in the original binary comparison script. The final model is
    a hinge-loss SVC, not the squared-residual refit in Mao's SVI method.
    The intercept is free in every base and final SVC.

    Parameters
    ----------
    M : int, default=5
        Number of base models. All base fits run sequentially.
    theta : float in (0, 1], default=0.1
        Fraction of supplied training rows in each base subset.
    sv_selection_ratio : float in (0, 1], default=0.3
        Fraction selected for the final fit. A selection with fewer than two
        rows or containing only one class raises ``IneligibleSelectionError``.
    C : float, default=1.0
        Positive native SVC penalty, shared by all base and final models.
        It is not divided by their sample sizes. In a mean hinge-loss
        objective the corresponding squared-norm coefficient is 1/(2*C*n).
    gamma : {"scale", "auto"} or positive float, default="scale"
        RBF coefficient, resolved once from the complete supplied training
        fold and then shared numerically by all learners. ``"scale"`` uses
        1/(n_features * X.var()); ``"auto"`` uses 1/n_features.
    kernel : {"rbf"}, default="rbf"
        This comparator currently supports the RBF kernel only.
    tol : float, default=1e-3
        Positive tolerance passed to every SVC optimizer.
    max_iter : int, default=-1
        Positive SVC iteration cap, or -1 for no cap. Nonconvergence retains
        SVC's warning behavior and is recorded in the fitted diagnostics.
    shrinking : bool, default=True
        Whether each SVC uses the shrinking heuristic.
    cache_size : float, default=200.0
        Kernel memory cache in MB for each active SVC fit. This is separate
        from the reusable disk cache specified by ``cache_dir``.
    random_state : int, RandomState, or None, default=42
        Private subset RNG. An input RandomState is copied, not consumed.
        None uses a fresh RNG and bypasses the reusable disk cache.
    prediction_batch_size : int or None, default=1024
        Maximum rows per decision or prediction call; None uses all rows.
        Revalidated at prediction time when changed after fitting.
    cache_dir : path-like or None, default=None
        Trusted private directory for checksummed base-stage results. Data,
        ordered subsets, parameters, and runtime identify entries. Changing
        only the selection ratio reuses the base stage but fits a new final
        model. Custom subclasses bypass caching. Failed computations are not
        cached; a successful base stage can survive a failed final refit.
        Reusing unconverged base models emits a ConvergenceWarning.
    store_base_estimators : bool, default=True
        Retain fitted base models in ``estimators_``; otherwise this is None.
        Consensus scores and diagnostics remain available in either case.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Ordered original labels; positive decision values refer to classes_[1].
    gamma_ : float
        Shared numeric RBF coefficient.
    subset_indices_ : list of M ndarrays
        Ordered training-row indices used by the base models.
    selected_indices_ : ndarray
        Ordered training-row indices used by the final SVC.
    consensus_scores_, selection_distances_ : ndarrays of shape (n_samples,)
        Raw mean base scores and their signed-margin selection distances.
    estimators_ : list or None
        Retained fitted base SVCs, according to store_base_estimators.
    final_estimator_ : SVC
        Fitted final classifier. Predictions use its native tie convention.
    n_iter_, fit_status_, converged_, support_counts_ : ndarrays of shape (M+1,)
        Base diagnostics followed by the final model; status zero indicates
        convergence and support_counts_ gives the total support-vector count.
    n_support_ : ndarray of shape (M+1, 2)
        Support-vector counts by ordered class for each base and final model.
    cache_hit_, cache_key_, cache_status_
        Reusable base-stage cache diagnostics.

    Notes
    -----
    This estimator performs no internal cross-validation or probability
    calibration. Put preprocessing and the complete ensemble in a Pipeline
    and tune it externally with GridSearchCV. Prefer parallel outer CV with
    sequential base fits; avoid multiplying outer workers by OvO pair workers.
    ``OvOClassifier(EnsembleKernelSVM(...))`` provides multiclass support.
    """

    def __init__(self, M=5, theta=0.1, sv_selection_ratio=0.3, C=1.0,
                 gamma="scale", kernel="rbf", tol=1e-3, max_iter=-1,
                 shrinking=True, cache_size=200.0, random_state=42,
                 prediction_batch_size=1024, cache_dir=None,
                 store_base_estimators=True):
        self.M = M
        self.theta = theta
        self.sv_selection_ratio = sv_selection_ratio
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.random_state = random_state
        self.prediction_batch_size = prediction_batch_size
        self.cache_dir = cache_dir
        self.store_base_estimators = store_base_estimators

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = False
        return tags

    def __sklearn_is_fitted__(self):
        return hasattr(self, "final_estimator_")

    def _clear_fit(self):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _validate_options(self):
        _positive_integer(self.M, "M")
        _fraction(self.theta, "theta")
        _fraction(self.sv_selection_ratio, "sv_selection_ratio")
        _finite_real(self.C, "C")
        _finite_real(self.tol, "tol")
        _finite_real(self.cache_size, "cache_size")
        if not isinstance(self.kernel, str) or self.kernel != "rbf":
            raise ValueError("kernel must be 'rbf'; other kernels are not supported.")
        if isinstance(self.gamma, str):
            if self.gamma not in ("scale", "auto"):
                raise ValueError("gamma must be 'scale', 'auto', or a finite positive number.")
        else:
            _finite_real(self.gamma, "gamma")
        if (isinstance(self.max_iter, (bool, np.bool_))
                or not isinstance(self.max_iter, Integral)
                or (self.max_iter != -1 and self.max_iter < 1)):
            raise ValueError("max_iter must be a positive integer or -1.")
        if not isinstance(self.shrinking, (bool, np.bool_)):
            raise TypeError("shrinking must be boolean.")
        if self.prediction_batch_size is not None:
            _positive_integer(self.prediction_batch_size, "prediction_batch_size")
        if self.cache_dir is not None and not isinstance(self.cache_dir, (str, PathLike)):
            raise TypeError("cache_dir must be a directory path or None.")
        if not isinstance(self.store_base_estimators, (bool, np.bool_)):
            raise TypeError("store_base_estimators must be boolean.")

    def _resolve_gamma(self, X):
        if isinstance(self.gamma, str):
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
            with np.errstate(over="ignore", invalid="ignore"):
                variance = float(X.var())
            if not np.isfinite(variance) or (variance == 0 and X.min() != X.max()):
                raise ValueError("Feature variance is outside float64 range; rescale X or specify gamma.")
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                gamma = (1.0 / X.shape[1]) / variance if variance != 0 else 1.0
            if not np.isfinite(gamma) or gamma <= 0:
                raise ValueError("Scaled gamma is outside float64 range; rescale X or specify gamma.")
            return gamma
        return float(self.gamma)

    def _make_estimator(self, gamma):
        return SVC(C=self.C, gamma=gamma, kernel=self.kernel, tol=self.tol,
                   max_iter=self.max_iter, shrinking=self.shrinking,
                   cache_size=self.cache_size, random_state=0)

    def fit(self, X, y):
        """Fit using only the supplied training rows; clear state on failure."""
        self._clear_fit()
        try:
            return self._fit(X, y)
        except Exception:
            self._clear_fit()
            raise

    def _base_stage(self, X, y, classes, subsets, gamma):
        models = []
        records = []
        score_sum = np.zeros(len(y), dtype=np.float64)
        for indices in subsets:
            model = self._make_estimator(gamma).fit(X[indices], y[indices])
            if not np.array_equal(model.classes_, classes):
                raise ValueError("All SVC learners must use the same ordered binary labels.")
            with np.errstate(over="ignore", invalid="ignore"):
                score_sum += _batched_scores(model, X, self.prediction_batch_size)
            models.append(model)
            records.append(_diagnostics(model))
        if np.isfinite(score_sum).all():
            consensus = score_sum / self.M
        else:
            try:
                consensus = _overflow_safe_consensus(models, X, self.prediction_batch_size)
            except FloatingPointError as error:
                raise FloatingPointError("SVC must return finite binary decision values.") from error
        signed_labels = np.where(y == classes[1], 1.0, -1.0)
        distances = np.abs(signed_labels * consensus - 1.0)
        return dict(consensus=consensus, distances=distances,
                    order=np.argsort(distances, kind="stable"),
                    models=models if self.store_base_estimators else None,
                    n_iter=np.asarray([item["n_iter"] for item in records], dtype=int),
                    fit_status=np.asarray([item["fit_status"] for item in records], dtype=int),
                    n_support=np.asarray([item["n_support"] for item in records], dtype=int))

    def _base_result(self, X, y, classes, subsets, gamma):
        compute = lambda: self._base_stage(X, y, classes, subsets, gamma)
        if self.cache_dir is None:
            reason = "disabled"
        elif self.random_state is None:
            reason = "bypassed: random_state=None"
        elif type(self) is not EnsembleKernelSVM:
            reason = "bypassed: custom estimator subclass"
        else:
            parameters = self.get_params(deep=False).copy()
            for name in ("sv_selection_ratio", "cache_dir", "random_state", "gamma"):
                parameters.pop(name)
            parameters["gamma"] = gamma
            key_data = dict(family="ensemble-kernel-svm-signed-margin-v1",
                            X=X, y=y, classes=classes, subsets=subsets,
                            parameters=parameters, provenance=_runtime_provenance(SVC))
            result, hit, key = _cached_base_stage(self.cache_dir, key_data, compute)
            if hit and np.any(result["fit_status"] != 0):
                count = int(np.count_nonzero(result["fit_status"]))
                warnings.warn(
                    f"Reusing {count} cached base SVC model(s) that did not converge. "
                    "Increase max_iter or change the parameter candidate.",
                    ConvergenceWarning, stacklevel=3)
            return result, hit, key, "hit" if hit else "miss"
        return compute(), False, None, reason

    def _fit(self, X, y):
        self._validate_options()
        X, y = validate_data(self, X, y, accept_sparse=False, dtype=np.float64)
        check_classification_targets(y)
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("EnsembleKernelSVM requires two classes; the input contains one class.")
        if len(classes) > 2:
            raise ValueError("Only binary classification is supported. The input has more than two classes.")
        selected_size = int(len(y) * self.sv_selection_ratio)
        if selected_size < 2:
            raise IneligibleSelectionError(
                "sv_selection_ratio selects fewer than two observations. "
                "Increase the ratio or provide more training observations.")
        subset_size = max(2, int(len(y) * self.theta))
        gamma = self._resolve_gamma(X)
        subsets = _sample_subsets(encoded, subset_size, self.M,
                                  _private_random_state(self.random_state))
        base, hit, key, status = self._base_result(X, y, classes, subsets, gamma)
        selected = base["order"][:selected_size].copy()
        if len(np.unique(encoded[selected])) != 2:
            raise IneligibleSelectionError(
                f"The {selected_size} selected observations contain only one class. "
                "This candidate cannot fit the final binary SVC. Increase "
                "sv_selection_ratio or use a different parameter candidate.")
        final = self._make_estimator(gamma).fit(X[selected], y[selected])
        if not np.array_equal(final.classes_, classes):
            raise ValueError("The final SVC must use the same ordered binary labels.")
        final_diagnostics = _diagnostics(final)
        self.classes_ = classes
        self.gamma_ = gamma
        self.subset_indices_ = subsets
        self.selected_indices_ = selected
        self.consensus_scores_ = base["consensus"]
        self.selection_distances_ = base["distances"]
        self.estimators_ = base["models"]
        self.n_iter_ = np.append(base["n_iter"], final_diagnostics["n_iter"])
        self.fit_status_ = np.append(base["fit_status"], final_diagnostics["fit_status"])
        self.converged_ = self.fit_status_ == 0
        self.n_support_ = np.vstack((base["n_support"], final_diagnostics["n_support"]))
        self.support_counts_ = self.n_support_.sum(axis=1)
        self.final_estimator_ = final
        self.cache_hit_ = hit
        self.cache_key_ = key
        self.cache_status_ = status
        return self

    def decision_function(self, X):
        """Return finite final-model scores, positive for classes_[1]."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64)
        return _batched_scores(self.final_estimator_, X, self.prediction_batch_size)

    def predict(self, X):
        """Return original labels using the final SVC's native tie convention."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64)
        size = len(X) if self.prediction_batch_size is None else _positive_integer(
            self.prediction_batch_size, "prediction_batch_size")
        result = np.empty(len(X), dtype=self.classes_.dtype)
        for start in range(0, len(X), size):
            stop = min(start + size, len(X))
            values = np.asarray(self.final_estimator_.predict(X[start:stop]))
            if values.shape != (stop - start,) or not np.isin(values, self.classes_).all():
                raise ValueError("SVC must return one valid class label per observation.")
            result[start:stop] = values
        return result
