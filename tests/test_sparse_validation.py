"""Unsupported sparse inputs must fail clearly without NumPy conversion."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from ensemble_dwd._validation import _reject_complex
from ensemble_dwd.estimator import _binary_scores
from ensemble_dwd.svm import _batched_scores


class NoArrayCSR(sparse.csr_array):
    def __array__(self, *args, **kwargs):
        raise AssertionError("Sparse validation must not call np.asarray")


class SparseScores:
    def __init__(self, values):
        self.values = values

    def decision_function(self, X):
        return self.values


class SparseValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
        self.y = np.array([0, 0, 1, 1])

    def containers(self, values):
        for form in ('csr', 'csc', 'coo', 'bsr', 'dia', 'dok', 'lil'):
            for kind in ('matrix', 'array'):
                factory = getattr(sparse, form + '_' + kind)
                yield factory(values)

    def models(self):
        options = dict(M=1, theta=1., sv_selection_ratio=1., random_state=1)
        return (EnsembleKernelDWD(**options, stopping='fixed', max_iter=3),
                EnsembleKernelSVM(**options), OvOClassifier(SVC()))

    def test_real_sparse_matrix_and_array_formats_reject_explicitly(self):
        for value in self.containers(self.X):
            with self.subTest(container=type(value).__name__):
                with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
                    _reject_complex(value, 'Features')

    def test_complex_sparse_dtype_still_rejects_as_complex(self):
        for imaginary in (0., 2.):
            for value in self.containers(self.X.astype(complex) + imaginary * 1j):
                with self.subTest(container=type(value).__name__, imaginary=imaginary):
                    with self.assertRaisesRegex(ValueError, '[Cc]omplex'):
                        _reject_complex(value, 'Features')

    def test_sparse_guard_never_calls_numpy_conversion(self):
        # NumPy 1.26/SciPy 1.12 can iterate older sparse arrays (including
        # dok_array) and raise NotImplementedError for 1D slices. Exercise it on
        # newer versions too, where that incidental conversion may succeed.
        value = NoArrayCSR(self.X)
        with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
            _reject_complex(value, 'Features')

    def test_sparse_first_fit_and_failed_refit_clear_learned_state(self):
        for value in self.containers(self.X):
            for model in self.models():
                with self.subTest(container=type(value).__name__, model=type(model).__name__):
                    for previous_fit in (False, True):
                        if previous_fit:
                            model.fit(self.X, self.y)
                        with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
                            model.fit(value, self.y)
                        with self.assertRaises(NotFittedError):
                            model.predict(self.X)
                        self.assertFalse(any(name.endswith('_') for name in vars(model)))

    def test_sparse_predictions_reject_and_preserve_existing_dense_fit(self):
        for model in self.models():
            model.fit(self.X, self.y)
            expected = model.predict(self.X)
            methods = [model.predict]
            methods.append(model.vote_counts if isinstance(model, OvOClassifier)
                           else model.decision_function)
            for value in self.containers(self.X):
                for method in methods:
                    with self.subTest(container=type(value).__name__, method=method.__name__):
                        with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
                            method(value)
            assert_array_equal(model.predict(self.X), expected)

    def test_sparse_labels_reject_before_label_conversion(self):
        for value in self.containers(self.y[:, None]):
            for model in self.models():
                with self.subTest(container=type(value).__name__, model=type(model).__name__):
                    with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
                        model.fit(self.X, value)

    def test_sparse_decision_values_are_explicitly_unsupported(self):
        for value in self.containers(self.y[:, None]):
            model = SparseScores(value)
            for action in (lambda: _binary_scores(model, self.X),
                           lambda: _batched_scores(model, self.X, None)):
                with self.subTest(container=type(value).__name__):
                    with self.assertRaisesRegex(TypeError, '[Ss]parse.*dense'):
                        action()


if __name__ == '__main__':
    unittest.main()
