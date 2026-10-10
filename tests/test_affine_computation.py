"""Optional joint affine scores across learners, fitted state and base caches."""

from fractions import Fraction
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier


class AffineComputationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack((np.arange(24, dtype=float) - 11.5,
                                  np.arange(24) % 3)) / 6.
        self.y = np.zeros(24, dtype=int)
        self.y[[10, 23]] = 1

    def model(self, **overrides):
        options = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
                       gamma=.2, max_iter=3, stopping='fixed', random_state=17)
        options.update(overrides)
        return EnsembleKernelDWD(**options)

    def assert_joint_learners(self, model):
        self.assertEqual(model.affine_computation_, 'joint')
        for learner in [*(model.estimators_ or []), model.final_estimator_]:
            self.assertEqual(learner.affine_computation, 'joint')
            self.assertEqual(learner.affine_computation_, 'joint')

    def test_standard_omits_new_keyword_for_older_dependencies(self):
        def older_constructor(**parameters):
            if 'affine_computation' in parameters:
                raise TypeError("unexpected keyword argument 'affine_computation'")
            return KernGDWD(**parameters)

        model = self.model()
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=older_constructor) as construct:
            model.fit(self.X, self.y)
        self.assertEqual(construct.call_count, model.M + 1)
        self.assertEqual(model.affine_computation_, 'standard')
        for call in construct.call_args_list:
            self.assertNotIn('affine_computation', call.kwargs)

    def test_joint_reaches_base_and_fresh_final_models_with_either_rbf_policy(self):
        for kernel, rbf in (('linear', 'standard'), ('rbf', 'standard'), ('rbf', 'direct')):
            with self.subTest(kernel=kernel, rbf=rbf):
                model = self.model(kernel=kernel, rbf_computation=rbf,
                                   affine_computation='joint').fit(self.X, self.y)
                self.assert_joint_learners(model)
                self.assertTrue(all(model.final_estimator_ is not base
                                    for base in model.estimators_))
                assert_array_equal(model.final_estimator_._Xfit, self.X[model.selected_indices_])
                expected = sum(base.decision_function(self.X) for base in model.estimators_) / model.M
                assert_array_equal(model.consensus_scores_, expected)
                assert_array_equal(model.selected_indices_, np.argsort(abs(expected), kind='stable'))
                combined = np.concatenate(model.subset_indices_)
                self.assertEqual(len(np.unique(combined)), len(combined))

    def test_joint_is_a_cloneable_nested_parameter(self):
        original = self.model(affine_computation='joint')
        copied = clone(original)
        self.assertEqual(copied.get_params(), original.get_params())
        copied.set_params(affine_computation='standard')
        self.assertEqual(original.affine_computation, 'joint')
        self.assertEqual(copied.affine_computation, 'standard')
        wrapper = clone(OvOClassifier(original))
        self.assertEqual(wrapper.get_params()['estimator__affine_computation'], 'joint')
        wrapper.set_params(estimator__affine_computation='standard')
        self.assertEqual(wrapper.estimator.affine_computation, 'standard')

    def test_older_serialized_models_keep_standard_scoring(self):
        model = self.model().fit(self.X, self.y)
        scores = model.decision_function(self.X)
        for learner in [model, *model.estimators_, model.final_estimator_]:
            learner.__dict__.pop('affine_computation', None)
            learner.__dict__.pop('affine_computation_', None)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.get_params()['affine_computation'], 'standard')
        self.assertEqual(clone(restored).affine_computation, 'standard')
        assert_array_equal(restored.decision_function(self.X), scores)

    def test_multiclass_forwards_joint_policy_into_all_real_pairs(self):
        labels = np.repeat(['a', 'b', 'c'], 8)
        wrapper = OvOClassifier(self.model(M=1, theta=1., affine_computation='joint'),
                                n_jobs=1).fit(self.X, labels)
        self.assertEqual(len(wrapper.estimators_), 3)
        for pair in wrapper.estimators_:
            self.assert_joint_learners(pair)
        self.assertEqual(wrapper.predict(self.X).shape, labels.shape)

    def test_invalid_policy_fails_before_construction_and_clears_refit(self):
        model = self.model(affine_computation='joint').fit(self.X, self.y)
        for policy in (None, True, 1, '', 'unknown', ['joint'], np.array(['joint'])):
            with self.subTest(policy=repr(policy)):
                model.set_params(affine_computation=policy)
                with patch('ensemble_dwd.estimator.KernGDWD') as construct:
                    with self.assertRaisesRegex(ValueError, 'affine_computation must be'):
                        model.fit(self.X, self.y)
                    construct.assert_not_called()
                self.assertFalse(any(name.endswith('_') for name in vars(model)))
                with self.assertRaises(NotFittedError):
                    model.predict(self.X)

    def test_joint_never_silently_falls_back_on_older_dependency(self):
        error = TypeError("unexpected keyword argument 'affine_computation'")
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=error) as construct:
            with self.assertRaisesRegex(ImportError, 'requested numerical computation options.*pinned') as caught:
                self.model(affine_computation='joint').fit(self.X, self.y)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(construct.call_count, 1)
        self.assertEqual(construct.call_args.kwargs['affine_computation'], 'joint')

    def test_cache_separates_policies_and_reuses_only_matching_bases(self):
        self.model(affine_computation='joint').fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            standard = self.model(cache_dir=directory).fit(self.X, self.y)
            joint = self.model(cache_dir=directory, affine_computation='joint').fit(self.X, self.y)
            self.assertFalse(standard.cache_hit_)
            self.assertFalse(joint.cache_hit_)
            self.assertNotEqual(standard.cache_key_, joint.cache_key_)
            for original in (standard, joint):
                repeated = clone(original)
                with patch.object(repeated, '_base_stage',
                                  side_effect=AssertionError('matching bases must be reused')):
                    repeated.fit(self.X, self.y)
                self.assertTrue(repeated.cache_hit_)
                self.assertEqual(repeated.cache_key_, original.cache_key_)
                self.assertIsNot(repeated.final_estimator_, original.final_estimator_)
                assert_array_equal(repeated.consensus_scores_, original.consensus_scores_)
                assert_array_equal(repeated.selected_indices_, original.selected_indices_)
                assert_array_equal(repeated.decision_function(self.X), original.decision_function(self.X))
                self.assertEqual(repeated.affine_computation_, original.affine_computation_)

    def test_fitted_joint_policy_survives_parameter_change_batching_and_pickle(self):
        model = self.model(affine_computation='joint').fit(self.X, self.y)
        expected_scores = model.decision_function(self.X)
        expected_labels = model.predict(self.X)
        model.set_params(affine_computation='standard')
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.affine_computation, 'standard')
        self.assertEqual(restored.affine_computation_, 'joint')
        self.assertEqual(restored.final_estimator_.affine_computation_, 'joint')
        assert_array_equal(restored.decision_function(self.X), expected_scores)
        assert_array_equal(restored.predict(self.X), expected_labels)
        # The ordinary accepted path need not be bitwise batch-invariant.
        linear = self.model(kernel='linear', affine_computation='joint').fit(self.X, self.y)
        query = np.array([[-.5, 0.], [0., .5], [1., -.5]])
        together = linear.decision_function(query)
        separately = np.concatenate([linear.decision_function(row[None, :]) for row in query])
        assert_allclose(together, separately, rtol=1e-14, atol=1e-14)
        assert_array_equal(together > 0, separately > 0)

    def test_real_two_row_fit_recovers_final_intercept_cancellation(self):
        X = np.array([[0.], [1.]])
        y = np.array([-3, 7])
        options = dict(M=1, theta=1., sv_selection_ratio=1., kernel='linear',
                       lambd=.1, max_iter=5, stopping='fixed', random_state=7)
        standard = EnsembleKernelDWD(**options).fit(X, y)
        joint = EnsembleKernelDWD(**options, affine_computation='joint').fit(X, y)
        query = np.array([[np.nextafter(.5, np.inf)]])
        model = joint.final_estimator_
        kernel = model._compute_kernel(query).T
        alpha = np.asarray(model.dual_coef_).ravel()
        intercept = float(np.asarray(model.intercept_).ravel()[0])
        exact = sum((Fraction.from_float(float(k)) * Fraction.from_float(float(a))
                     for k, a in zip(kernel[0], alpha)), Fraction.from_float(intercept))
        self.assertGreater(exact, 0)
        self.assertEqual(joint.decision_function(query)[0], float(exact))
        self.assertEqual(joint.predict(query)[0], 7)
        self.assertEqual(standard.decision_function(query)[0], 0.)
        self.assertEqual(standard.predict(query)[0], -3)
        repeated = np.repeat(query, 3, axis=0)
        assert_array_equal(joint.decision_function(repeated), np.repeat(float(exact), 3))
        assert_array_equal(joint.final_estimator_.dual_coef_, standard.final_estimator_.dual_coef_)
        assert_array_equal(joint.final_estimator_.intercept_, standard.final_estimator_.intercept_)


if __name__ == '__main__':
    unittest.main()
