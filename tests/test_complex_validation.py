"""Complex values must never become real data through an implicit cast."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from ensemble_dwd._validation import _reject_complex
from ensemble_dwd.estimator import _binary_scores
from ensemble_dwd.svm import _batched_scores

EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
from tune_mnist import select_pair
from tune_mnist_multiclass import validate_split


def complex_variants(values):
    for imaginary in (0., 1.):
        yield np.asarray(values, dtype=complex) + imaginary * 1j
        for scalar in (complex, np.complex128):
            array = np.empty(np.shape(values), dtype=object)
            for index in np.ndindex(array.shape):
                array[index] = scalar(values[index] + imaginary * 1j)
            yield array
        nested = np.empty(np.shape(values), dtype=object)
        for index in np.ndindex(nested.shape):
            nested[index] = np.array(np.complex128(values[index] + imaginary * 1j), dtype=object)
        yield nested


class ScoreModel:
    def __init__(self, values):
        self.values = values

    def decision_function(self, X):
        return self.values[:len(X)]

    def predict(self, X):
        return self.values[:len(X)]


class ComplexValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
        self.y = np.array([0, 0, 1, 1])

    def assert_rejected_without_warning(self, call):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(ValueError, "[Cc]omplex"):
                call()

    def models(self):
        options = dict(M=1, theta=1., sv_selection_ratio=1., random_state=1)
        return (EnsembleKernelDWD(**options), EnsembleKernelSVM(**options),
                OvOClassifier(SVC()))

    def test_complex_features_and_labels_rejected_before_fit(self):
        for values, target in ((self.X, "features"), (self.y, "labels")):
            for invalid in complex_variants(values):
                for model in self.models():
                    with self.subTest(model=type(model).__name__, target=target, dtype=invalid.dtype):
                        self.assert_rejected_without_warning(
                            lambda: model.fit(invalid if target == "features" else self.X,
                                              invalid if target == "labels" else self.y))
                        self.assertFalse(any(name.endswith("_") for name in vars(model)))

    def test_failed_refit_discards_previous_state(self):
        for model in self.models():
            model.classes_ = np.array([0, 1])
            model.n_features_in_ = 2
            model.final_estimator_ = ScoreModel(self.y)
            invalid = next(complex_variants(self.X))
            self.assert_rejected_without_warning(lambda: model.fit(invalid, self.y))
            with self.assertRaises(NotFittedError):
                model.predict(self.X)

    def test_prediction_features_rejected_before_underlying_prediction(self):
        for model in self.models():
            model.classes_ = np.array([0, 1])
            model.n_features_in_ = 2
            model.final_estimator_ = ScoreModel(self.y)
            model.estimators_ = [ScoreModel(self.y)]
            for invalid in complex_variants(self.X):
                self.assert_rejected_without_warning(lambda: model.predict(invalid))

    def test_complex_decision_values_rejected_before_cast(self):
        values = np.array([-1., 0., .5, 1.])
        for invalid in complex_variants(values):
            model = ScoreModel(invalid)
            self.assert_rejected_without_warning(lambda: _binary_scores(model, self.X))
            self.assert_rejected_without_warning(lambda: _batched_scores(model, self.X, 2))

    def test_real_decision_values_preserve_exact_values(self):
        values = np.array([-1., 0., .5, 1.])
        for dtype in (np.float64, np.float32, object):
            model = ScoreModel(values.astype(dtype))
            assert_array_equal(_binary_scores(model, self.X), values)
            assert_array_equal(_batched_scores(model, self.X, None), values)

    def test_complex_label_outputs_do_not_become_labels_or_votes(self):
        for invalid in complex_variants(self.y):
            svm = EnsembleKernelSVM()
            svm.classes_ = np.array([0, 1])
            svm.n_features_in_ = 2
            svm.final_estimator_ = ScoreModel(invalid)
            self.assert_rejected_without_warning(lambda: svm.predict(self.X))
        ovo = OvOClassifier(SVC()).fit(self.X, self.y)
        for invalid in complex_variants(np.array([-1, -1, 1, 1])):
            ovo.estimators_ = [ScoreModel(invalid)]
            self.assert_rejected_without_warning(lambda: ovo.predict(self.X))

    def test_mnist_labels_rejected_before_conversion(self):
        for invalid in complex_variants(self.y):
            self.assert_rejected_without_warning(lambda: select_pair(self.X, invalid, 0, 1))
        for invalid in complex_variants(np.arange(10)):
            self.assert_rejected_without_warning(
                lambda: validate_split(np.ones((10, 2)), invalid, "supplied split"))

    def test_mnist_real_object_labels_preserve_existing_results(self):
        X, y = select_pair(self.X, self.y.astype(object), 0, 1)
        assert_array_equal(X, self.X)
        assert_array_equal(y, self.y)
        _, y = validate_split(np.ones((10, 2)), np.arange(10).astype(object), "split")
        assert_array_equal(y, np.arange(10))

    def test_object_array_cycles_do_not_recurse_forever(self):
        array = np.empty(2, dtype=object)
        array[0], array[1] = array, 1.
        self.assertIsNone(_reject_complex(array, "test"))
        array[1] = np.complex128(1j)
        self.assert_rejected_without_warning(lambda: _reject_complex(array, "test"))

    def test_deep_object_arrays_reject_without_recursive_calls(self):
        value = np.array(np.complex128(1j), dtype=object)
        for _ in range(32):
            parent = np.empty((), dtype=object)
            parent[()] = value
            value = parent
        self.assert_rejected_without_warning(lambda: _reject_complex(value, "test"))


if __name__ == "__main__":
    unittest.main()
