"""Regressions for representable consensus means and scaled-kernel limits."""

from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from ensemble_dwd import EnsembleKernelDWD


class SequenceDWD:
    """Return predetermined finite base margins independently of fitted rows."""

    margins = []
    fitted = 0

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.index = type(self).fitted
        type(self).fitted += 1
        self.classes_ = np.unique(y)
        self.n_iter_ = 1
        self.converged_ = True
        return self

    def decision_function(self, X):
        value = self.margins[self.index] if self.index < len(self.margins) else 0.
        return np.full(len(X), value, dtype=np.float64)


class NumericalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-2., 2., 10)[:, None]
        self.y = np.tile([0, 1], 5)

    def fit_margins(self, margins):
        SequenceDWD.margins = margins
        SequenceDWD.fitted = 0
        model = EnsembleKernelDWD(M=len(margins), theta=.2,
                                  sv_selection_ratio=1., gamma=.1)
        with patch('ensemble_dwd.estimator.KernGDWD', SequenceDWD):
            model.fit(self.X, self.y)
        self.assertEqual(SequenceDWD.fitted, len(margins) + 1)
        return model

    def test_finite_mean_survives_same_sign_sum_overflow(self):
        largest = np.finfo(np.float64).max
        for values in ([1e308, 1e308], [largest, largest], [-largest, -largest]):
            with self.subTest(values=values):
                model = self.fit_margins(values)
                expected = float(sum(Fraction.from_float(float(x)) for x in values) / len(values))
                assert_array_equal(model.consensus_scores_, np.full(len(self.y), expected))
                self.assertTrue(np.isfinite(model.consensus_scores_).all())
                assert_array_equal(model.selected_indices_, np.arange(len(self.y)))

    def test_overflow_recovery_preserves_opposing_large_terms_and_small_remainder(self):
        largest = np.finfo(np.float64).max
        values = [largest, largest, -largest, -largest, 1.]
        model = self.fit_margins(values)
        assert_array_equal(model.consensus_scores_, np.full(len(self.y), .2))
        assert_array_equal(model.selected_indices_, np.arange(len(self.y)))

    def test_ordinary_consensus_arithmetic_remains_unchanged(self):
        values = [.3, -.7, .1, 1.1, -.2]
        model = self.fit_margins(values)
        expected = np.mean(np.repeat(np.asarray(values)[:, None], len(self.y), axis=1), axis=0)
        assert_array_equal(model.consensus_scores_, expected)

    def test_nonconstant_variance_underflow_is_not_treated_as_constant_input(self):
        X = self.X * 1e-200
        self.assertEqual(float(X.var()), 0.)
        self.assertGreater(np.unique(X).size, 1)
        model = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1.)
        with patch('ensemble_dwd.estimator.KernGDWD', SequenceDWD):
            with self.assertRaisesRegex(ValueError, '(?i)underflow'):
                model.fit(X, self.y)
        self.assertFalse(hasattr(model, 'final_estimator_'))

    def test_truly_constant_tiny_input_keeps_scale_gamma_one(self):
        SequenceDWD.margins = [0.]
        SequenceDWD.fitted = 0
        model = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1.)
        with patch('ensemble_dwd.estimator.KernGDWD', SequenceDWD):
            model.fit(np.full((len(self.y), 1), 1e-200), self.y)
        self.assertEqual(model.gamma_, 1.)
        self.assertEqual(model.final_estimator_.params['implementation'], 'optimized')


if __name__ == '__main__':
    unittest.main()
