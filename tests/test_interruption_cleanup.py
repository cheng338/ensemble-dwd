"""Interruption cleanup uses deterministic stand-ins, without optimizer fits."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier
from ensemble_dwd import multiclass


class ToyBinary(ClassifierMixin, BaseEstimator):
    prediction_calls = 0

    def __init__(self, scalar_iterations=False):
        self.scalar_iterations = scalar_iterations

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_iter_ = 1 if self.scalar_iterations else np.array([1])
        self.converged_ = True
        self.fit_status_ = 0
        self.n_support_ = np.array([1, 1])
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, 0]

    def predict(self, X):
        type(self).prediction_calls += 1
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class InterruptionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.], [-1.], [1.], [2.]])
        self.y = np.array([-1, -1, 1, 1])
        ToyBinary.prediction_calls = 0

    def assert_clean(self, model, prediction_calls):
        self.assertEqual([name for name in vars(model) if name.endswith('_')], [])
        self.assertFalse(model.__sklearn_is_fitted__())
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)
        self.assertEqual(ToyBinary.prediction_calls, prediction_calls)

    def test_binary_interruptions_clear_old_and_partial_fitted_state(self):
        for cls in (EnsembleKernelDWD, EnsembleKernelSVM):
            for exception in (KeyboardInterrupt, SystemExit):
                with self.subTest(estimator=cls.__name__, exception=exception.__name__):
                    model = cls(M=1, theta=1., sv_selection_ratio=1., gamma=.1)
                    with patch.object(model, '_make_estimator', side_effect=lambda *args: ToyBinary(
                            scalar_iterations=cls is EnsembleKernelDWD)):
                        model.fit(self.X, self.y)
                    self.assertEqual(model.predict(self.X).shape, self.y.shape)
                    calls = ToyBinary.prediction_calls
                    interruption = exception('preserve this interruption')

                    def interrupt(*args):
                        # Real input validation has already written this attribute.
                        self.assertTrue(hasattr(model, 'n_features_in_'))
                        raise interruption

                    with patch.object(model, '_base_result', side_effect=interrupt):
                        with self.assertRaises(exception) as caught:
                            model.fit(self.X, self.y)
                    self.assertIs(caught.exception, interruption)
                    self.assert_clean(model, calls)

    def test_ovo_interruptions_clear_old_and_partial_fitted_state(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__):
                model = OvOClassifier(ToyBinary(), n_jobs=1).fit(self.X, self.y)
                self.assertEqual(model.predict(self.X).shape, self.y.shape)
                calls = ToyBinary.prediction_calls
                interruption = exception('preserve this interruption')

                def interrupt(*args):
                    self.assertTrue(hasattr(model, 'n_features_in_'))
                    raise interruption

                with patch.object(multiclass, '_fit_pair', side_effect=interrupt):
                    with self.assertRaises(exception) as caught:
                        model.fit(self.X, self.y)
                self.assertIs(caught.exception, interruption)
                self.assert_clean(model, calls)


if __name__ == '__main__':
    unittest.main()
