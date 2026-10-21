"""Automatic restart forwarding, persisted policy and compact diagnostics."""
import pickle
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from ensemble_dwd import EnsembleKernelDWD


class AccelerationDefaultTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(21620).normal(size=(16, 3))
        self.y = np.tile([-3, 7], 8)
        self.options = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.1,
                            gamma=.3, max_iter=3, stopping='fixed', random_state=2)

    def test_auto_forwards_concrete_compatible_policy(self):
        for backend in ('auto', 'spectral', 'lbfgs'):
            effective = None if backend == 'lbfgs' else 'restart'
            a = EnsembleKernelDWD(**self.options, backend=backend).fit(self.X, self.y)
            b = EnsembleKernelDWD(**self.options, backend=backend, acceleration=effective).fit(self.X, self.y)
            for name in ('subset_indices_', 'selected_indices_', 'consensus_scores_'):
                assert_array_equal(getattr(a, name), getattr(b, name))
            for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                assert_array_equal(getattr(a.final_estimator_, name), getattr(b.final_estimator_, name))
            assert_array_equal(a.decision_function(self.X), b.decision_function(self.X))
            self.assertEqual(a.requested_acceleration_, 'auto')
            self.assertEqual(a.acceleration_, effective)
            for learner in a.estimators_ + [a.final_estimator_]:
                self.assertEqual(learner.acceleration, effective)

    def test_old_missing_field_and_fitted_pickle(self):
        old = EnsembleKernelDWD(**self.options)
        del old.acceleration
        restored = pickle.loads(pickle.dumps(old))
        self.assertIsNone(clone(restored).acceleration)
        new = EnsembleKernelDWD(**self.options).fit(self.X, self.y)
        copied = pickle.loads(pickle.dumps(new))
        copied.set_params(acceleration=None)
        assert_array_equal(copied.predict(self.X), new.predict(self.X))
        self.assertEqual(copied.requested_acceleration_, 'auto')
        self.assertEqual(copied.acceleration_, 'restart')

    def test_lean_warm_cache_keeps_concrete_base_diagnostics_and_requested_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(self.options, store_base_estimators=False, cache_dir=directory)
            a = EnsembleKernelDWD(**options).fit(self.X, self.y)
            warm = EnsembleKernelDWD(**options).fit(self.X, self.y)
            plain = EnsembleKernelDWD(**options, acceleration=None).fit(self.X, self.y)
            explicit = EnsembleKernelDWD(**options, acceleration='restart').fit(self.X, self.y)
            self.assertFalse(a.cache_hit_)
            self.assertTrue(warm.cache_hit_)
            self.assertFalse(plain.cache_hit_)
            self.assertFalse(explicit.cache_hit_)
            self.assertEqual(len({a.cache_key_, plain.cache_key_, explicit.cache_key_}), 3)
            for model in (a, warm):
                self.assertEqual(model.requested_acceleration_, 'auto')
                self.assertEqual(model.acceleration_, 'restart')
                for diagnostic in model.base_diagnostics_:
                    self.assertEqual(diagnostic['solver_diagnostics']['requested_acceleration'], 'restart')
                    self.assertEqual(diagnostic['solver_diagnostics']['acceleration'], 'restart')
                    self.assertFalse(any(isinstance(value, np.ndarray)
                                         for value in diagnostic['solver_diagnostics'].values()))


if __name__ == '__main__': unittest.main()
