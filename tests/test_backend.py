"""Backend forwarding, cloning, validation, and cache identity contracts."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ensemble_dwd import EnsembleKernelDWD, OvOClassifier


class BackendProbeDWD:
    """Record construction and fit calls without running a numerical solver."""

    instances = []
    fits = []
    failure = None

    def __init__(self, **params):
        self.params = params
        type(self).instances.append(self)

    def fit(self, X, y):
        type(self).fits.append(self)
        if type(self).failure is not None:
            raise type(self).failure
        self.classes_ = np.unique(y)
        self.n_iter_ = 2
        self.converged_ = True
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, 0]

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class BackendTests(unittest.TestCase):
    def setUp(self):
        BackendProbeDWD.instances = []
        BackendProbeDWD.fits = []
        BackendProbeDWD.failure = None
        self.X = np.column_stack((np.linspace(-2., 2., 24), np.arange(24) % 3))
        self.y = np.tile(['left', 'right'], 12)

    def model(self, **overrides):
        params = dict(M=2, theta=.5, sv_selection_ratio=1., gamma=.2,
                      random_state=17)
        params.update(overrides)
        return EnsembleKernelDWD(**params)

    def test_default_and_explicit_backends_reach_every_learner(self):
        for backend in ('auto', 'spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                BackendProbeDWD.instances = []
                model = self.model() if backend == 'auto' else self.model(backend=backend)
                with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                    model.fit(self.X, self.y)
                self.assertEqual(len(BackendProbeDWD.instances), model.M + 1)
                for learner in [*model.estimators_, model.final_estimator_]:
                    self.assertEqual(learner.params['backend'], backend)
                    self.assertEqual(learner.params['implementation'], 'optimized')
                    self.assertEqual(learner.params['solver_mode'], 'schur')
                    self.assertEqual(learner.params['initialization'], 'zero')
                    self.assertEqual(learner.params['lambd'], model.lambd)
                    self.assertEqual(learner.params['kernel_kws'], {'gamma': .2})
                    self.assertEqual(learner.params['max_iter'], 100)
                    self.assertEqual(learner.params['stopping'], 'objective')

    def test_real_dependency_constructor_receives_backend_without_fitting(self):
        for backend in ('auto', 'spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                learner = self.model(backend=backend)._make_estimator({'gamma': .2})
                self.assertEqual(learner.get_params()['backend'], backend)
                self.assertEqual(learner.get_params()['implementation'], 'optimized')

    def test_backend_is_a_cloneable_public_parameter(self):
        self.assertEqual(EnsembleKernelDWD().get_params()['backend'], 'auto')
        original = self.model(backend='spectral')
        copied = clone(original)
        self.assertEqual(copied.get_params(), original.get_params())
        self.assertEqual(copied.backend, 'spectral')
        self.assertIn("backend='spectral'", repr(copied))
        self.assertIs(copied.set_params(backend='cholesky'), copied)
        self.assertEqual(copied.backend, 'cholesky')
        self.assertEqual(original.backend, 'spectral')

    def test_invalid_backend_fails_before_constructing_learners(self):
        for backend in (None, True, 1, '', 'eigen', 'AUTO', ['auto'], np.array(['auto'])):
            with self.subTest(backend=repr(backend)):
                with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                    with self.assertRaisesRegex(ValueError, 'backend must be'):
                        self.model(backend=backend).fit(self.X, self.y)
                self.assertEqual(BackendProbeDWD.instances, [])

    def test_invalid_backend_refit_clears_fitted_state(self):
        model = self.model(backend='spectral')
        with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
            model.fit(self.X, self.y)
            model.set_params(backend='invalid')
            with self.assertRaisesRegex(ValueError, 'backend must be'):
                model.fit(self.X, self.y)
        self.assertEqual(model.backend, 'invalid')
        self.assertFalse(any(name.endswith('_') for name in vars(model)))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_backend_fit_failure_propagates_and_clears_fitted_state(self):
        model = self.model(backend='spectral')
        failure = FloatingPointError('injected backend failure')
        with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
            model.fit(self.X, self.y)
            BackendProbeDWD.failure = failure
            with self.assertRaises(FloatingPointError) as caught:
                model.fit(self.X, self.y)
        self.assertIs(caught.exception, failure)
        self.assertFalse(any(name.endswith('_') for name in vars(model)))
        self.assertEqual(model.backend, 'spectral')

    def test_cache_separates_backends_and_reuses_matching_bases(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                first = self.model(cache_dir=cache_dir).fit(self.X, self.y)
                spectral = self.model(cache_dir=cache_dir, backend='spectral').fit(self.X, self.y)
                repeated = self.model(cache_dir=cache_dir, backend='spectral').fit(self.X, self.y)
                returned = self.model(cache_dir=cache_dir).fit(self.X, self.y)
        self.assertFalse(first.cache_hit_)
        self.assertFalse(spectral.cache_hit_)
        self.assertNotEqual(first.cache_key_, spectral.cache_key_)
        self.assertTrue(repeated.cache_hit_)
        self.assertTrue(returned.cache_hit_)
        self.assertEqual(repeated.cache_key_, spectral.cache_key_)
        self.assertEqual(returned.cache_key_, first.cache_key_)
        self.assertEqual(len(BackendProbeDWD.fits), 2 * (first.M + 1) + 2)
        self.assertIsNot(repeated.final_estimator_, spectral.final_estimator_)
        self.assertIsNot(returned.final_estimator_, first.final_estimator_)
        self.assertEqual(repeated.final_estimator_.params['backend'], 'spectral')
        self.assertEqual(returned.final_estimator_.params['backend'], 'auto')

    def test_multiclass_clones_and_forwards_backend_to_each_pair(self):
        wrapper = OvOClassifier(self.model(), n_jobs=1)
        wrapper.set_params(estimator__backend='spectral')
        copied = clone(wrapper)
        self.assertEqual(copied.get_params()['estimator__backend'], 'spectral')
        y = np.repeat(['a', 'b', 'c'], 8)
        with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
            copied.fit(self.X, y)
        self.assertEqual(len(copied.estimators_), 3)
        self.assertEqual(len(BackendProbeDWD.fits), 3 * (self.model().M + 1))
        for pair in copied.estimators_:
            self.assertEqual(pair.backend, 'spectral')
            for learner in [*pair.estimators_, pair.final_estimator_]:
                self.assertEqual(learner.params['backend'], 'spectral')

    def test_residual_check_order_clones_and_reaches_all_learners(self):
        self.assertEqual(self.model().get_params()['residual_check_order'], 'auto')
        for order in ('refinement_first', 'adaptive'):
            with self.subTest(order=order):
                original = self.model(residual_check_order=order)
                model = clone(original)
                self.assertEqual(model.get_params(), original.get_params())
                with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                    model.fit(self.X, self.y)
                for learner in [*model.estimators_, model.final_estimator_]:
                    self.assertEqual(learner.params['residual_check_order'], order)
                real = original._make_estimator({'gamma': .2})
                self.assertEqual(real.get_params()['residual_check_order'], order)

    def test_invalid_residual_check_order_fails_before_constructing_learners(self):
        for order in (None, True, 1, '', 'automatic', ['adaptive'], np.array(['adaptive'])):
            with self.subTest(order=repr(order)), patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                with self.assertRaisesRegex(ValueError, 'residual_check_order must be'):
                    self.model(residual_check_order=order).fit(self.X, self.y)
                self.assertEqual(BackendProbeDWD.instances, [])
        for options in (dict(backend='spectral'), dict(backend='lbfgs')):
            with self.subTest(options=options), patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
                with self.assertRaisesRegex(ValueError, 'optimized Cholesky'):
                    self.model(residual_check_order='adaptive', **options).fit(self.X, self.y)
                self.assertEqual(BackendProbeDWD.instances, [])

    def test_restart_adaptive_order_reaches_every_learner_and_clones(self):
        original = self.model(residual_check_order='adaptive', acceleration='restart')
        model = clone(original)
        self.assertEqual(model.get_params(), original.get_params())
        with patch('ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
            model.fit(self.X, self.y)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner.params['residual_check_order'], 'adaptive')
            self.assertEqual(learner.params['acceleration'], 'restart')

    def test_real_restart_adaptive_fit_preserves_healthy_ensemble(self):
        options = dict(M=2, theta=.25, sv_selection_ratio=1., random_state=5,
                       acceleration='restart', max_iter=8, stopping='fixed')
        default = EnsembleKernelDWD(**options).fit(self.X, self.y)
        adaptive = EnsembleKernelDWD(**options, residual_check_order='adaptive').fit(self.X, self.y)
        np.testing.assert_array_equal(default.selected_indices_, adaptive.selected_indices_)
        np.testing.assert_array_equal(default.predict(self.X), adaptive.predict(self.X))
        for first, second in zip([*default.estimators_, default.final_estimator_],
                                 [*adaptive.estimators_, adaptive.final_estimator_]):
            np.testing.assert_array_equal(first.dual_coef_, second.dual_coef_)
            np.testing.assert_array_equal(first.objective_history_, second.objective_history_)
            self.assertEqual(second.diagnostics_['residual_check_order'], 'adaptive')
            self.assertEqual(second.diagnostics_['acceleration'], 'restart')

    def test_check_order_separates_cache_and_retention_still_reuses_bases(self):
        with tempfile.TemporaryDirectory() as cache_dir, patch(
                'ensemble_dwd.estimator.KernGDWD', BackendProbeDWD):
            ordinary = self.model(cache_dir=cache_dir).fit(self.X, self.y)
            adaptive = self.model(cache_dir=cache_dir, residual_check_order='adaptive').fit(self.X, self.y)
            repeated = self.model(cache_dir=cache_dir, residual_check_order='adaptive',
                                  sv_selection_ratio=.75).fit(self.X, self.y)
        self.assertFalse(ordinary.cache_hit_)
        self.assertFalse(adaptive.cache_hit_)
        self.assertNotEqual(ordinary.cache_key_, adaptive.cache_key_)
        self.assertTrue(repeated.cache_hit_)
        self.assertEqual(repeated.cache_key_, adaptive.cache_key_)
        self.assertEqual(len(BackendProbeDWD.fits), 2 * (ordinary.M + 1) + 1)


if __name__ == '__main__':
    unittest.main()
