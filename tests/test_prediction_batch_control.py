"""Final prediction follows the wrapper's current memory bound without mutation."""

import pickle
from inspect import signature
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ensemble_dwd import EnsembleKernelDWD


class OlderChild:
    def __init__(self):
        self.prediction_batch_size = 1
        self.calls = []

    def decision_function(self, X):
        self.calls.append(len(X))
        return X[:, 0]


class OverrideChild(OlderChild):
    def decision_function(self, X, *, prediction_batch_size="omitted"):
        self.calls.append((len(X), prediction_batch_size))
        return X[:, 0]


def wrapper(child, batch_size):
    model = EnsembleKernelDWD(prediction_batch_size=batch_size)
    model.classes_ = np.array(["a", "b"])
    model.n_features_in_ = 1
    model.final_estimator_ = child
    return model


class PredictionBatchControlTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(7, dtype=float).reshape(-1, 1) - 3

    def test_explicit_override_and_none_do_not_modify_child_setting(self):
        child = OverrideChild()
        model = wrapper(child, 3)
        assert_array_equal(model.decision_function(self.X), self.X[:, 0])
        model.set_params(prediction_batch_size=None)
        assert_array_equal(model.decision_function(self.X), self.X[:, 0])
        model.set_params(prediction_batch_size=2)
        assert_array_equal(model.predict(self.X), np.where(self.X[:, 0] > 0, "b", "a"))
        self.assertEqual(child.calls, [(7, 3), (7, None), (7, 2)])
        self.assertEqual(child.prediction_batch_size, 1)

    def test_older_or_custom_child_receives_bounded_rows_without_new_keyword(self):
        child = OlderChild()
        model = wrapper(child, 3)
        assert_array_equal(model.decision_function(self.X), self.X[:, 0])
        self.assertEqual(child.calls, [3, 3, 1])
        self.assertEqual(child.prediction_batch_size, 1)
        child.calls.clear()
        model.set_params(prediction_batch_size=None)
        assert_array_equal(model.decision_function(self.X), self.X[:, 0])
        self.assertEqual(child.calls, [7])
        self.assertEqual(child.prediction_batch_size, 1)

    def test_invalid_postfit_setting_is_rejected_before_any_child_call(self):
        for child in (OlderChild(), OverrideChild()):
            model = wrapper(child, 3)
            for invalid in (0, -1, True, np.bool_(False), 1.5, "2", []):
                with self.subTest(child=type(child).__name__, invalid=invalid):
                    model.prediction_batch_size = invalid
                    with self.assertRaisesRegex(ValueError, "prediction_batch_size"):
                        model.decision_function(self.X)
                    self.assertEqual(child.calls, [])

    def test_real_child_typeerror_is_not_retried_as_an_older_interface(self):
        class FailingChild(OverrideChild):
            def decision_function(self, X, *, prediction_batch_size=None):
                self.calls.append(prediction_batch_size)
                raise TypeError("intentional scoring error")
        child = FailingChild()
        with self.assertRaisesRegex(TypeError, "intentional scoring error"):
            wrapper(child, 3).decision_function(self.X)
        self.assertEqual(child.calls, [3])
        self.assertEqual(child.prediction_batch_size, 1)

    def test_invalid_child_scores_remain_rejected_in_compatibility_batches(self):
        class InvalidChild(OlderChild):
            def decision_function(self, X):
                return np.full(len(X), np.nan)
        with self.assertRaisesRegex(FloatingPointError, "finite decision"):
            wrapper(InvalidChild(), 3).decision_function(self.X)

    def test_real_final_model_uses_updated_batching_and_preserves_fitted_state(self):
        model = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., lambd=.1,
            gamma=.5, max_iter=3, stopping="fixed", prediction_batch_size=1,
            random_state=7).fit(self.X, np.where(self.X[:, 0] > 0, 1, -1))
        child = model.final_estimator_
        per_call_override = "prediction_batch_size" in signature(child.decision_function).parameters
        before = pickle.dumps(child)
        reference = model.decision_function(self.X)
        for setting, expected in ((3, [3, 3, 1]), (None, [7]), (2, [2, 2, 2, 1])):
            model.prediction_batch_size = setting
            calls = []
            original = child._compute_kernel

            def capture(X):
                calls.append(len(X))
                return original(X)

            with patch.object(child, "_compute_kernel", side_effect=capture):
                actual = model.decision_function(self.X)
            # Released older DWD retains its original tighter child limit.
            # New DWD must honor the override, including explicit None.
            self.assertEqual(calls, expected if per_call_override else [1] * len(self.X))
            assert_allclose(actual, reference, atol=1e-14, rtol=1e-14)
            self.assertEqual(pickle.dumps(child), before)
            self.assertEqual(child.prediction_batch_size, 1)


if __name__ == "__main__":
    unittest.main()
