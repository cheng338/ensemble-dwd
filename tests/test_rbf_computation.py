"""Optional RBF arithmetic, dependency compatibility, and cache contracts."""

import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier


class RBFComputationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack((np.arange(24, dtype=float) - 11.5,
                                  np.arange(24) % 3)) / 6.
        self.y = np.zeros(24, dtype=int)
        # Each seed-17 disjoint block contains a rare-class observation.
        self.y[[10, 23]] = 1

    def model(self, **overrides):
        options = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
                       gamma=.2, max_iter=3, stopping='fixed', random_state=17)
        options.update(overrides)
        return EnsembleKernelDWD(**options)

    def assert_direct_learners(self, model):
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner.rbf_computation, 'direct')
            self.assertEqual(learner.kernel_computation_, 'direct_v1')
            self.assertEqual(learner.kernel_kws, {'gamma': model.gamma_})

    def test_standard_omits_the_new_keyword_for_older_dependencies(self):
        def older_constructor(**parameters):
            if 'rbf_computation' in parameters:
                raise TypeError("unexpected keyword argument 'rbf_computation'")
            return KernGDWD(**parameters)

        model = self.model()
        self.assertEqual(model.rbf_computation, 'standard')
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=older_constructor) as construct:
            model.fit(self.X, self.y)
        self.assertEqual(construct.call_count, model.M + 1)
        for call in construct.call_args_list:
            self.assertNotIn('rbf_computation', call.kwargs)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertIn(learner.kernel_computation_, ('sklearn', 'norm_sum'))

    def test_direct_reaches_real_base_and_fresh_final_learners(self):
        model = self.model(rbf_computation='direct').fit(self.X, self.y)
        self.assert_direct_learners(model)
        self.assertEqual(len(model.estimators_), model.M)
        self.assertTrue(all(model.final_estimator_ is not base for base in model.estimators_))
        assert_array_equal(model.final_estimator_._Xfit, self.X[model.selected_indices_])
        combined = np.concatenate(model.subset_indices_)
        self.assertEqual(len(np.unique(combined)), len(combined))

    def test_cloning_nested_parameters_and_older_serialized_objects(self):
        original = self.model(rbf_computation='direct')
        copied = clone(original)
        self.assertEqual(copied.get_params(), original.get_params())
        copied.set_params(rbf_computation='standard')
        self.assertEqual(original.rbf_computation, 'direct')
        self.assertEqual(copied.rbf_computation, 'standard')
        wrapped = clone(OvOClassifier(original))
        self.assertEqual(wrapped.get_params()['estimator__rbf_computation'], 'direct')
        wrapped.set_params(estimator__rbf_computation='standard')
        self.assertEqual(wrapped.estimator.rbf_computation, 'standard')

        old = self.model().fit(self.X, self.y)
        scores = old.decision_function(self.X)
        del old.__dict__['rbf_computation']
        for learner in [*old.estimators_, old.final_estimator_]:
            del learner.__dict__['rbf_computation']
        restored = pickle.loads(pickle.dumps(old))
        self.assertEqual(restored.get_params()['rbf_computation'], 'standard')
        self.assertEqual(clone(restored).rbf_computation, 'standard')
        assert_array_equal(restored.decision_function(self.X), scores)

    def test_multiclass_clones_direct_policy_into_every_real_pair(self):
        labels = np.repeat(['a', 'b', 'c'], 8)
        binary = self.model(M=1, theta=1., rbf_computation='direct')
        wrapper = OvOClassifier(binary, n_jobs=1).fit(self.X, labels)
        self.assertEqual(len(wrapper.estimators_), 3)
        for pair in wrapper.estimators_:
            self.assertEqual(pair.rbf_computation, 'direct')
            self.assert_direct_learners(pair)
        self.assertEqual(wrapper.predict(self.X).shape, labels.shape)

    def test_invalid_policy_is_rejected_before_constructing_learners(self):
        for policy in (None, True, 1, '', 'unknown', ['direct'], np.array(['direct'])):
            with self.subTest(policy=repr(policy)):
                with patch('ensemble_dwd.estimator.KernGDWD') as construct:
                    with self.assertRaisesRegex(ValueError, 'rbf_computation must be'):
                        self.model(rbf_computation=policy).fit(self.X, self.y)
                    construct.assert_not_called()

    def test_direct_rejects_non_rbf_kernels_before_constructing_learners(self):
        for kernel in ('linear', 'poly', 'precomputed', lambda X, Y: X @ Y.T):
            with self.subTest(kernel=repr(kernel)):
                with patch('ensemble_dwd.estimator.KernGDWD') as construct:
                    with self.assertRaisesRegex(ValueError, 'requires the named RBF'):
                        self.model(kernel=kernel, rbf_computation='direct').fit(self.X, self.y)
                    construct.assert_not_called()

    def test_unsupported_direct_keywords_leave_failed_refit_unfitted(self):
        model = self.model(rbf_computation='direct').fit(self.X, self.y)
        model.set_params(kernel_kws={'degree': 2})
        with self.assertRaisesRegex(ValueError, 'kernel keywords'):
            model.fit(self.X, self.y)
        self.assertFalse(any(name.endswith('_') for name in vars(model)))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_cache_separates_policies_and_replays_only_matching_bases(self):
        # Initialize numerical modules before the cache's guarded observations.
        self.model(rbf_computation='direct').fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            standard = self.model(cache_dir=directory).fit(self.X, self.y)
            direct = self.model(cache_dir=directory, rbf_computation='direct').fit(self.X, self.y)
            self.assertFalse(standard.cache_hit_)
            self.assertFalse(direct.cache_hit_)
            self.assertNotEqual(standard.cache_key_, direct.cache_key_)

            repeated = self.model(cache_dir=directory, rbf_computation='direct')
            with patch.object(repeated, '_base_stage', side_effect=AssertionError('base stage must be reused')):
                repeated.fit(self.X, self.y)
            returned = self.model(cache_dir=directory)
            with patch.object(returned, '_base_stage', side_effect=AssertionError('base stage must be reused')):
                returned.fit(self.X, self.y)
            self.assertTrue(repeated.cache_hit_)
            self.assertTrue(returned.cache_hit_)
            self.assertEqual(repeated.cache_key_, direct.cache_key_)
            self.assertEqual(returned.cache_key_, standard.cache_key_)
            self.assertIsNot(repeated.final_estimator_, direct.final_estimator_)
            self.assertIsNot(returned.final_estimator_, standard.final_estimator_)
            self.assert_direct_learners(repeated)
            for original, replayed in ((direct, repeated), (standard, returned)):
                assert_array_equal(original.consensus_scores_, replayed.consensus_scores_)
                assert_array_equal(original.selected_indices_, replayed.selected_indices_)
                assert_array_equal(original.final_estimator_.dual_coef_, replayed.final_estimator_.dual_coef_)
                assert_array_equal(original.final_estimator_.intercept_, replayed.final_estimator_.intercept_)
                assert_array_equal(original.decision_function(self.X), replayed.decision_function(self.X))
                assert_array_equal(original.predict(self.X), replayed.predict(self.X))

    def test_fitted_prediction_policy_survives_parameter_change_and_pickle(self):
        from dwd._direct_rbf import accurate_rbf

        model = self.model(rbf_computation='direct').fit(self.X, self.y)
        query = self.X[[1, 6, 8, 15]].copy()
        expected_kernel = accurate_rbf(model.final_estimator_._Xfit, query, gamma=model.gamma_)
        assert_array_equal(model.final_estimator_._compute_kernel(query), expected_kernel)
        expected_scores = model.decision_function(query)
        expected_labels = model.predict(query)
        model.set_params(rbf_computation='standard')
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.rbf_computation, 'standard')
        self.assertEqual(restored.final_estimator_.kernel_computation_, 'direct_v1')
        assert_array_equal(restored.final_estimator_._compute_kernel(query), expected_kernel)
        assert_array_equal(restored.decision_function(query), expected_scores)
        assert_array_equal(restored.predict(query), expected_labels)

    def test_explicit_direct_does_not_fall_back_on_an_older_dependency(self):
        error = TypeError("unexpected keyword argument 'rbf_computation'")
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=error) as construct:
            with self.assertRaises(ImportError) as caught:
                self.model(rbf_computation='direct')._make_estimator({'gamma': .2})
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(construct.call_count, 1)
        self.assertEqual(construct.call_args.kwargs['rbf_computation'], 'direct')


if __name__ == '__main__':
    unittest.main()
