"""Global class-orientation changes preserve the specified selection geometry."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd.svm import EnsembleKernelSVM


class SelectionOrientationTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=1)
        self.addCleanup(self.limits.restore_original_limits)
        rng = np.random.default_rng(181025)
        self.y = np.tile([-3, 7], 16)
        self.reversed_y = np.where(self.y == 7, -3, 7)
        self.X = rng.normal(size=(32, 4))
        self.X[:, 0] += np.where(self.y == 7, .6, -.6)
        self.subsets = [np.arange(8), np.arange(8, 16)]

    def test_dwd_global_orientation_preserves_absolute_consensus_selection(self):
        models = []
        for y in (self.y, self.reversed_y):
            with patch('ensemble_dwd.estimator._sample_subsets', return_value=self.subsets):
                model = EnsembleKernelDWD(M=2, theta=.25, sv_selection_ratio=.75,
                    lambd=.05, gamma=.2, max_iter=8, stopping='fixed',
                    prediction_batch_size=5, random_state=18).fit(self.X, y)
            models.append(model)
        normal, reversed_model = models
        assert_allclose(reversed_model.consensus_scores_, -normal.consensus_scores_, rtol=0, atol=1e-12)
        expected = sorted(range(len(self.X)), key=lambda i: (abs(normal.consensus_scores_[i]), i))[:24]
        assert_array_equal(normal.selected_indices_, expected)
        assert_array_equal(reversed_model.selected_indices_, expected)
        assert_array_equal(normal.final_estimator_._Xfit, self.X[expected])
        self.assertIsNot(normal.final_estimator_, normal.estimators_[0])
        scores = normal.decision_function(self.X)
        opposite = reversed_model.decision_function(self.X)
        assert_allclose(opposite, -scores, rtol=0, atol=1e-12)
        separated = abs(scores) > 1e-10
        self.assertTrue(separated.any())
        assert_array_equal(reversed_model.predict(self.X)[separated],
                           np.where(normal.predict(self.X)[separated] == 7, -3, 7))

    def test_svm_global_orientation_preserves_signed_margin_selection(self):
        models = []
        for y in (self.y, self.reversed_y):
            with patch('ensemble_dwd.svm._sample_subsets', return_value=self.subsets):
                model = EnsembleKernelSVM(M=2, theta=.25, sv_selection_ratio=.75,
                    C=.7, gamma=.2, tol=1e-7, random_state=18).fit(self.X, y)
            models.append(model)
        normal, reversed_model = models
        assert_allclose(reversed_model.consensus_scores_, -normal.consensus_scores_, rtol=0, atol=2e-6)
        signed = np.where(self.y == 7, 1., -1.)
        distance = abs(signed * normal.consensus_scores_ - 1.)
        expected = sorted(range(len(self.X)), key=lambda i: (distance[i], i))[:24]
        assert_allclose(normal.selection_distances_, distance, rtol=0, atol=0)
        assert_array_equal(normal.selected_indices_, expected)
        assert_array_equal(reversed_model.selected_indices_, expected)
        # This fixture distinguishes the implemented signed-margin distance
        # from the unsigned distance to either margin boundary.
        unsigned = sorted(range(len(self.X)),
                          key=lambda i: (abs(abs(normal.consensus_scores_[i]) - 1.), i))[:24]
        self.assertNotEqual(expected, unsigned)
        self.assertIsNot(normal.final_estimator_, normal.estimators_[0])


if __name__ == '__main__':
    unittest.main()
