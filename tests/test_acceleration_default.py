"""Automatic restart forwarding, persisted policy and compact diagnostics."""
from copy import deepcopy
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD


class AccelerationDefaultTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(21620).normal(size=(16, 3))
        self.y = np.tile([-3, 7], 8)
        self.options = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.1,
                            gamma=.3, max_iter=3, stopping='fixed', random_state=2)

    def _fit_with_requests(self, model):
        requests = []
        original_fit = KernGDWD.fit

        def record_fit(learner, X, y, **fit_options):
            requests.append((learner, deepcopy(learner.get_params(deep=False)),
                             X.copy(), y.copy(), deepcopy(fit_options)))
            return original_fit(learner, X, y, **fit_options)

        # Observe the real fit boundary; do not replace the constructor,
        # eigensolver, validators, factor construction or numerical solver.
        with patch.object(KernGDWD, 'fit', record_fit):
            model.fit(self.X, self.y)
        return model, requests

    def _check_requests(self, model, requests, requested, effective):
        self.assertEqual(model.requested_acceleration_, requested)
        self.assertEqual(model.acceleration_, effective)
        learners = model.estimators_ + [model.final_estimator_]
        rows = list(model.subset_indices_) + [model.selected_indices_]
        self.assertEqual(len(requests), model.M + 1)
        self.assertEqual(len(learners), len(requests))
        self.assertEqual(len({id(learner) for learner in learners}), model.M + 1)
        self.assertEqual(len(rows), len(requests))
        for (learner, parameters, X, y, fit_options), fitted, indices in zip(
                requests, learners, rows):
            self.assertIs(learner, fitted)
            self.assertEqual(parameters['acceleration'], effective)
            self.assertEqual(parameters['backend'], model.backend)
            self.assertEqual(fitted.acceleration, effective)
            self.assertEqual(fitted.acceleration_, effective)
            assert_array_equal(X, self.X[indices])
            assert_array_equal(y, self.y[indices])
            self.assertEqual(fit_options, {})

    def test_auto_forwards_concrete_compatible_policy(self):
        for backend in ('auto', 'spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                effective = None if backend == 'lbfgs' else 'restart'
                a, a_requests = self._fit_with_requests(
                    EnsembleKernelDWD(**self.options, backend=backend))
                b, b_requests = self._fit_with_requests(EnsembleKernelDWD(
                    **self.options, backend=backend, acceleration=effective))
                self._check_requests(a, a_requests, 'auto', effective)
                self._check_requests(b, b_requests, effective, effective)
                # Equivalent policy requests must delegate identical options
                # and sampled base data, without assuming independent native
                # eigendecompositions produce bitwise identical fitted arrays.
                assert_array_equal(a.subset_indices_, b.subset_indices_)
                for first, second in zip(a_requests, b_requests):
                    self.assertEqual(first[1], second[1])
                for first, second in zip(a_requests[:-1], b_requests[:-1]):
                    assert_array_equal(first[2], second[2])
                    assert_array_equal(first[3], second[3])
                # Selection order can change with tiny score differences.
                # Check each final fit against its own selected rows above.
                # This fixture selects all rows.
                for model in (a, b):
                    assert_array_equal(np.sort(model.selected_indices_),
                                       np.arange(len(self.X)))
                    assert_array_equal(model.classes_, np.unique(self.y))
                    assert_array_equal(model.predict(self.X),
                                       model.final_estimator_.predict(self.X))
                    self.assertTrue(np.isfinite(model.consensus_scores_).all())
                    scores = model.decision_function(self.X)
                    self.assertEqual(scores.shape, (len(self.X),))
                    self.assertTrue(np.isfinite(scores).all())

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
