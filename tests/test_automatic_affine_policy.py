"""Resolve affine policy across dependency versions and ensemble learners."""
import pickle
import tempfile
import unittest
from inspect import signature
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.cv import run_cv
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier

class AutomaticAffinePolicyTests(unittest.TestCase):

    def setUp(self):
        self.limits = threadpool_limits(1)
        self.addCleanup(self.limits.restore_original_limits)
        self.X = np.array([[0.0], [1.0]])
        self.y = np.array([-3, 7])
        self.query = np.array([[np.nextafter(0.5, np.inf)]])
        self.Xn = np.column_stack((np.linspace(-2, 2, 24), np.sin(np.linspace(-2, 2, 24))))
        self.yn = np.tile([-1, 1], 12)

    def ensemble(self, **kw):
        options = dict(M=2, theta=0.5, sv_selection_ratio=1.0, lambd=0.05, gamma=0.2, max_iter=3, stopping='fixed', random_state=17)
        options.update(kw)
        return EnsembleKernelDWD(**options)

    def test_ensemble_default_all_learners_and_nested_ovo(self):
        auto = self.ensemble().fit(self.Xn, self.yn)
        joint = self.ensemble(affine_computation='joint').fit(self.Xn, self.yn)
        self.assertEqual(auto.affine_computation, 'auto')
        self.assertEqual(auto.affine_computation_, 'joint')
        for model in [*auto.estimators_, auto.final_estimator_]:
            self.assertEqual(model.affine_computation, 'joint')
            self.assertEqual(model.affine_computation_, 'joint')
        np.testing.assert_array_equal(auto.selected_indices_, joint.selected_indices_)
        np.testing.assert_array_equal(auto.consensus_scores_, joint.consensus_scores_)
        labels = np.repeat(['a', 'b', 'c'], 8)
        wrapper = OvOClassifier(self.ensemble(M=1, theta=1.0), n_jobs=1).fit(self.Xn, labels)
        self.assertEqual(clone(wrapper).get_params()['estimator__affine_computation'], 'auto')
        self.assertTrue(all((m.affine_computation_ == 'joint' for m in wrapper.estimators_)))

    def test_ensemble_explicit_standard_with_new_dependency(self):
        model = self.ensemble(affine_computation='standard').fit(self.Xn, self.yn)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner.affine_computation, 'standard')
            self.assertEqual(learner.affine_computation_, 'standard')

    def test_ensemble_unknown_wrapper_conservative_without_retry(self):
        calls = []

        def hidden_constructor(**kw):
            calls.append(kw.copy())
            self.assertNotIn('affine_computation', kw)
            return KernGDWD(**kw)
        with patch('ensemble_dwd.estimator.KernGDWD', new=hidden_constructor):
            model = self.ensemble().fit(self.Xn, self.yn)
        self.assertEqual(len(calls), model.M + 1)
        self.assertEqual(model.affine_computation_, 'standard')
        self.assertTrue(all((m.affine_computation_ == 'standard' for m in model.estimators_)))

    def test_dependency_internal_typeerror_is_not_retried(self):
        error = TypeError('internal constructor failure')
        calls = []

        def broken(*, affine_computation='standard', **kw):
            calls.append(affine_computation)
            raise error
        with patch('ensemble_dwd.estimator.KernGDWD', new=broken):
            with self.assertRaises(ImportError) as caught:
                self.ensemble().fit(self.Xn, self.yn)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(calls, ['joint'])

    def test_ensemble_fitted_mutation_pickle_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.ensemble(cache_dir=directory).fit(self.Xn, self.yn)
            repeat = clone(original).fit(self.Xn, self.yn)
            self.assertTrue(repeat.cache_hit_)
            standard = self.ensemble(cache_dir=directory, affine_computation='standard').fit(self.Xn, self.yn)
            self.assertFalse(standard.cache_hit_)
            self.assertNotEqual(original.cache_key_, standard.cache_key_)
            scores = original.decision_function(self.Xn.copy())
            original.set_params(affine_computation='standard')
            restored = pickle.loads(pickle.dumps(original))
            self.assertEqual(restored.affine_computation_, 'joint')
            np.testing.assert_array_equal(restored.decision_function(self.Xn.copy()), scores)
            restored.fit(self.Xn, self.yn)
            self.assertEqual(restored.affine_computation_, 'standard')
