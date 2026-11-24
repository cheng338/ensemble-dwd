"""One-versus-one classification with the dissertation's maximum-wins rule."""

from itertools import combinations
from mmap import mmap
from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from joblib.parallel import get_active_backend
from sklearn.base import BaseEstimator, clone
from dwd._validation import ExactLabelClassifierMixin, _discrete_training_labels, exact_sparse_uint64_target
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted, validate_data

from .estimator import IneligibleSelectionError, IneligibleSubsetError, _positive_integer
from ._failure_context import copy_failure_context
from ._validation import _class_mask, _label_value, _reject_complex


def _pair_context(pair):
    labels = tuple(map(_label_value, pair))
    return f"OVO pair {labels!r}"


def _backing_memmap(array):
    """Find the NumPy map joblib would reopen, including through ndarray views."""
    base = array
    seen = set()
    while (parent := getattr(base, "base", None)) is not None and not isinstance(parent, mmap):
        if id(base) in seen:
            return None
        seen.add(id(base))
        base = parent
    return base if isinstance(base, np.memmap) and parent is not None else None


def _needs_snapshot(array, backing):
    if backing is None:
        return False
    if backing.mode == "c":
        return True
    if array.flags.c_contiguous or array.flags.f_contiguous or min(array.strides) >= 0:
        return False
    return any(stride < 0 and size > 1 for stride, size in zip(array.strides, array.shape))


def _pair_worker_inputs(X, y, n_jobs):
    """Preserve the parent's feature rows and labels across joblib dispatch."""
    if n_jobs == 1 or (X.base is None and y.base is None):
        return X, y, False
    X_base = _backing_memmap(X)
    y_base = _backing_memmap(y)
    copy_X = _needs_snapshot(X, X_base)
    copy_y = _needs_snapshot(y, y_base)
    transpose = False
    if X_base is not None and not copy_X and X.flags.c_contiguous != X.flags.f_contiguous:
        # joblib #1934 selects the backing map's order for contiguous views.
        file_is_fortran = X_base.ndim > 1 and X_base.flags.f_contiguous
        transpose = X.flags.f_contiguous != file_is_fortran
    if not (copy_X or copy_y or transpose):
        return X, y, False
    backend, _ = get_active_backend()
    if getattr(backend, "supports_sharedmem", False):
        return X, y, False
    if n_jobs is None:
        n_jobs = Parallel().n_jobs
    if backend.effective_n_jobs(n_jobs) == 1:
        return X, y, False
    # Snapshot parent-private data and reversed views once before the pair loop.
    if copy_y:
        y = np.array(y, copy=True, order="K")
    if copy_X:
        X = np.array(X, copy=True, order="K")
    return X.T if transpose else X, y, transpose


def _fit_pair(estimator, X, y, indices, pair, transpose=False):
    """Materialize only the pair being fitted in the current worker."""
    try:
        if transpose:
            X = X.T
        model = clone(estimator)
        pair_y = np.where(_class_mask(y[indices], pair[0]), 1, -1)
        model.fit(X[indices], pair_y)
        return model
    except IneligibleSubsetError as exc:
        wrapped = IneligibleSubsetError(f"{_pair_context(pair)}: {exc}")
        try:
            copy_failure_context(exc, wrapped)
        except Exception:
            pass
        raise wrapped from exc
    except IneligibleSelectionError as exc:
        wrapped = IneligibleSelectionError(f"{_pair_context(pair)}: {exc}")
        try:
            copy_failure_context(exc, wrapped)
        except Exception:
            pass
        raise wrapped from exc
    except Exception as exc:
        exc.add_note(f"While fitting {_pair_context(pair)}.")
        raise


class OvOClassifier(ExactLabelClassifierMixin, BaseEstimator):
    """Fit one binary estimator per class pair and predict by maximum wins.

    This implements the multiclass extension in Chang Cheng's dissertation:
    fit K * (K - 1) / 2 independent binary classifiers, give each pair one vote,
    and predict the class receiving the most votes. Ties choose the first class
    in sorted ``classes_`` order.
    Decision-score magnitudes do not change this voting rule.

    Parameters
    ----------
    estimator : estimator object
        Binary classifier with ``fit`` and ``predict``. Use an
        ``EnsembleKernelDWD`` instance for ensemble kernel DWD, or an
        ``EnsembleKernelSVM`` instance for the ensemble SVM comparator. A fresh clone
        is fitted for every pair with the first original class encoded as +1
        and the second as -1. In particular, an exactly zero DWD margin votes
        for the second original class.
    n_jobs : nonzero int or None, default=1
        Number of pair-fitting workers, using joblib's usual convention.
        None means one worker unless a joblib parallel context overrides it;
        -1 uses all available workers. Keep this at 1 when parallelizing outer
        cross-validation to avoid multiplying concurrent kernel fits. Pair
        matrices are created within their worker, with at most one queued
        pair per worker. Prediction runs sequentially over pairs in batches.
    prediction_batch_size : positive int, default=1024
        Maximum number of query rows passed to a binary classifier at once.
        Voting requires only this many rows by K classes during ``predict``;
        it never constructs an n_samples by n_pairs prediction matrix.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted original class labels, with their original dtype.
    estimators_ : list of length n_classes * (n_classes - 1) / 2
        Fitted binary classifiers in lexicographic class-pair order.
    pairs_ : tuple of label pairs
        Original labels corresponding to each fitted binary classifier.
    class_pair_indices_ : ndarray of shape (n_pairs, 2)
        Positions in ``classes_`` corresponding to ``pairs_``.
    pair_indices_ : tuple of integer ndarrays
        Training-row positions used for each binary fit, relative to the
        input supplied to this estimator. A child's subset/selection indices
        are relative to its pair; index this array to recover original rows.
    n_features_in_ : int
        Number of input features seen during fit.
    feature_names_in_ : ndarray, when input has all-string feature names
        Feature names seen during fit.

    Notes
    -----
    All pairs clone the same parameter configuration, including its random
    state. No cross-validation is performed here. Use a preprocessing Pipeline
    and external GridSearchCV to select one shared configuration across all
    pairs, with nested parameter names such as ``estimator__lambd`` for DWD
    or ``estimator__C`` for SVM.

    A numeric kernel gamma is shared across pairs. Both ``EnsembleKernelDWD``
    and ``EnsembleKernelSVM`` resolve gamma='scale' from each pair's input
    separately, before sampling their base learners. Use numeric gamma for an
    identical coefficient across all pairwise classifiers. The wrapper leaves
    each binary estimator's objective, intercept handling, selection rule and
    stopping policy unchanged. DWD uses absolute consensus scores and its MM
    controls; SVM uses signed-margin selection and native SVC controls.

    Inputs are dense, finite feature rows converted to float64. Precomputed
    kernels are not supported because rows are selected for each pair.
    Failed fits propagate their errors, with pair context, and leave this
    estimator unfitted; no fallback classifier is substituted.
    """

    def __init__(self, estimator, n_jobs=1, prediction_batch_size=1024):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.prediction_batch_size = prediction_batch_size

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        tags.input_tags.sparse = False
        return tags

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimators_")

    def _clear_fit(self):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _validate_options(self):
        if (self.n_jobs is not None
                and (isinstance(self.n_jobs, (bool, np.bool_))
                     or not isinstance(self.n_jobs, Integral) or self.n_jobs == 0)):
            raise ValueError("n_jobs must be a nonzero integer or None.")
        _positive_integer(self.prediction_batch_size, "prediction_batch_size")
        if not all(callable(getattr(self.estimator, name, None))
                   for name in ("fit", "predict")):
            raise TypeError("estimator must provide callable fit and predict methods.")
        if get_tags(self.estimator).input_tags.pairwise:
            raise ValueError("OvOClassifier does not support precomputed kernels.")

    def fit(self, X, y):
        """Fit a fresh binary estimator for every observed class pair."""
        return self._fit(X, y)

    def _fit(self, X, y, *, pair_dispatch=None):
        """Fit with an explicit internal dispatcher, preserving pair semantics."""
        self._clear_fit()
        try:
            self._validate_options()
            _reject_complex(X, "Features")
            y = exact_sparse_uint64_target(y)
            _reject_complex(y, "Labels")
            X, y = validate_data(self, X, y, accept_sparse=False, dtype=np.float64)
            y = _discrete_training_labels(y)
            classes = np.unique(y)
            if len(classes) < 2:
                raise ValueError("OvOClassifier requires at least 2 classes; got 1 class.")
            class_pairs = np.asarray(list(combinations(range(len(classes)), 2)),
                                     dtype=np.intp)
            pairs = tuple((classes[first], classes[second])
                          for first, second in class_pairs)
            indices = tuple(np.flatnonzero(_class_mask(y, first) | _class_mask(y, second))
                            for first, second in pairs)
            if pair_dispatch is None:
                parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch="n_jobs", batch_size=1)
                # Parallel resolves inherited counts and converts them to integers.
                worker_X, worker_y, transpose = _pair_worker_inputs(X, y, parallel.n_jobs)
                models = parallel(
                    delayed(_fit_pair)(self.estimator, worker_X, worker_y, rows, pair, transpose)
                    for rows, pair in zip(indices, pairs)
                )
            else:
                models = pair_dispatch(self.estimator, X, y, indices, pairs)
            self.classes_ = classes
            self.pairs_ = pairs
            self.class_pair_indices_ = class_pairs
            self.pair_indices_ = indices
            self.estimators_ = models
            return self
        except BaseException:
            self._clear_fit()
            raise

    def _prediction_data(self, X):
        check_is_fitted(self)
        _positive_integer(self.prediction_batch_size, "prediction_batch_size")
        _reject_complex(X, "Features")
        return validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64)

    def _batch_votes(self, X):
        votes = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(X))
        for model, pair, (first, second) in zip(
                self.estimators_, self.pairs_, self.class_pair_indices_):
            try:
                predictions = np.asarray(model.predict(X))
                _reject_complex(predictions, "Binary predicted labels")
                if predictions.shape != (len(X),):
                    raise ValueError("A binary estimator must predict one label per row.")
                if not np.isin(predictions, (-1, 1)).all():
                    raise ValueError("A binary estimator must predict labels -1 or +1.")
                winners = np.where(predictions == 1, first, second)
                votes[rows, winners] += 1
            except Exception as exc:
                exc.add_note(f"While predicting {_pair_context(pair)}.")
                raise
        return votes

    def vote_counts(self, X):
        """Return integer pairwise win counts in sorted ``classes_`` order.

        The result has shape (n_samples, n_classes). Each row sums to the
        number of fitted class pairs. Counts are votes, not probabilities.
        """
        X = self._prediction_data(X)
        result = np.empty((len(X), len(self.classes_)), dtype=np.int64)
        for start in range(0, len(X), self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, len(X))
            result[start:stop] = self._batch_votes(X[start:stop])
        return result

    def predict(self, X):
        """Predict by maximum wins; exact vote ties choose the smallest class."""
        X = self._prediction_data(X)
        result = np.empty(len(X), dtype=self.classes_.dtype)
        for start in range(0, len(X), self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, len(X))
            votes = self._batch_votes(X[start:stop])
            result[start:stop] = self.classes_[np.argmax(votes, axis=1)]
        return result
