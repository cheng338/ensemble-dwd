"""Optional exact-recovery budgets must remain distinct through ensemble fits."""

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


class ExactRecoveryTests(unittest.TestCase):
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

    def assert_extended_learners(self, model):
        self.assertEqual(model.exact_recovery_, 'extended')
        for learner in [*(model.estimators_ or []), model.final_estimator_]:
            self.assertEqual(learner.exact_recovery, 'extended')
            self.assertEqual(learner.exact_recovery_, 'extended')
        self.assertEqual([record['exact_recovery'] for record in model.base_diagnostics_],
                         ['extended'] * model.M)

    def test_standard_omits_new_keyword_for_older_dependencies(self):
        def older_constructor(**parameters):
            if 'exact_recovery' in parameters:
                raise TypeError("unexpected keyword argument 'exact_recovery'")
            return KernGDWD(**parameters)

        model = self.model()
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=older_constructor) as construct:
            model.fit(self.X, self.y)
        self.assertEqual(construct.call_count, model.M + 1)
        self.assertEqual(model.exact_recovery_, 'standard')
        for call in construct.call_args_list:
            self.assertNotIn('exact_recovery', call.kwargs)

    def test_extended_reaches_real_base_and_fresh_final_fits(self):
        for options in ({'kernel': 'linear'}, {'kernel': 'rbf'},
                        {'rbf_computation': 'direct', 'affine_computation': 'joint'}):
            with self.subTest(options=options):
                model = self.model(exact_recovery='extended', **options).fit(self.X, self.y)
                self.assert_extended_learners(model)
                self.assertTrue(all(model.final_estimator_ is not base
                                    for base in model.estimators_))
                assert_array_equal(model.final_estimator_._Xfit, self.X[model.selected_indices_])
                expected = sum(base.decision_function(self.X) for base in model.estimators_) / model.M
                assert_array_equal(model.consensus_scores_, expected)
                assert_array_equal(model.selected_indices_, np.argsort(abs(expected), kind='stable'))
                combined = np.concatenate(model.subset_indices_)
                self.assertEqual(len(np.unique(combined)), len(combined))

    def test_ordinary_fit_states_and_predictions_are_unchanged(self):
        standard = self.model().fit(self.X, self.y)
        extended = self.model(exact_recovery='extended').fit(self.X, self.y)
        for old, new in zip([*standard.estimators_, standard.final_estimator_],
                            [*extended.estimators_, extended.final_estimator_]):
            assert_array_equal(old.dual_coef_, new.dual_coef_)
            assert_array_equal(old.intercept_, new.intercept_)
            assert_array_equal(old.objective_history_, new.objective_history_)
            self.assertEqual(old.n_iter_, new.n_iter_)
            self.assertEqual(old.termination_reason_, new.termination_reason_)
        assert_array_equal(standard.consensus_scores_, extended.consensus_scores_)
        assert_array_equal(standard.selected_indices_, extended.selected_indices_)
        assert_array_equal(standard.decision_function(self.X), extended.decision_function(self.X))
        assert_array_equal(standard.predict(self.X), extended.predict(self.X))

    def test_clone_nested_parameter_and_real_multiclass_pairs(self):
        original = self.model(M=1, theta=1., exact_recovery='extended')
        copied = clone(original)
        self.assertEqual(copied.get_params(), original.get_params())
        copied.set_params(exact_recovery='standard')
        self.assertEqual(original.exact_recovery, 'extended')
        self.assertEqual(copied.exact_recovery, 'standard')
        wrapper = clone(OvOClassifier(original, n_jobs=1))
        self.assertEqual(wrapper.get_params()['estimator__exact_recovery'], 'extended')
        labels = np.repeat(['a', 'b', 'c'], 8)
        wrapper.fit(self.X, labels)
        self.assertEqual(len(wrapper.estimators_), 3)
        for pair in wrapper.estimators_:
            self.assert_extended_learners(pair)
        self.assertEqual(wrapper.predict(self.X).shape, labels.shape)

    def test_invalid_policy_fails_before_construction_and_clears_refit(self):
        model = self.model(exact_recovery='extended').fit(self.X, self.y)
        for policy in (None, True, 1, '', 'unknown', ['extended'], np.array(['extended'])):
            with self.subTest(policy=repr(policy)):
                model.set_params(exact_recovery=policy)
                with patch('ensemble_dwd.estimator.KernGDWD') as construct:
                    with self.assertRaisesRegex(ValueError, 'exact_recovery must be'):
                        model.fit(self.X, self.y)
                    construct.assert_not_called()
                self.assertFalse(any(name.endswith('_') for name in vars(model)))
                with self.assertRaises(NotFittedError):
                    model.predict(self.X)

    def test_extended_never_falls_back_on_older_dependency(self):
        error = TypeError("unexpected keyword argument 'exact_recovery'")
        with patch('ensemble_dwd.estimator.KernGDWD', side_effect=error) as construct:
            with self.assertRaisesRegex(ImportError, 'requested numerical computation options.*pinned') as caught:
                self.model(exact_recovery='extended').fit(self.X, self.y)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(construct.call_count, 1)
        self.assertEqual(construct.call_args.kwargs['exact_recovery'], 'extended')

    def test_provenance_survives_lean_storage_parameter_change_and_pickle(self):
        model = self.model(exact_recovery='extended', store_base_estimators=False).fit(self.X, self.y)
        self.assertIsNone(model.estimators_)
        self.assert_extended_learners(model)
        scores = model.decision_function(self.X)
        model.set_params(exact_recovery='standard')
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.exact_recovery, 'standard')
        self.assert_extended_learners(restored)
        assert_array_equal(restored.decision_function(self.X), scores)
        restored.fit(self.X, self.y)
        self.assertEqual(restored.exact_recovery_, 'standard')
        self.assertEqual(restored.final_estimator_.exact_recovery_, 'standard')

    def test_old_serialized_models_keep_standard_constructor_behavior(self):
        model = self.model().fit(self.X, self.y)
        scores = model.decision_function(self.X)
        for learner in [model, *model.estimators_, model.final_estimator_]:
            learner.__dict__.pop('exact_recovery', None)
            learner.__dict__.pop('exact_recovery_', None)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.get_params()['exact_recovery'], 'standard')
        self.assertEqual(clone(restored).exact_recovery, 'standard')
        assert_array_equal(restored.decision_function(self.X), scores)

    def test_cache_separates_policies_and_replays_only_matching_bases(self):
        self.model(exact_recovery='extended').fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            standard = self.model(cache_dir=directory).fit(self.X, self.y)
            extended = self.model(cache_dir=directory, exact_recovery='extended').fit(self.X, self.y)
            self.assertFalse(standard.cache_hit_)
            self.assertFalse(extended.cache_hit_)
            self.assertNotEqual(standard.cache_key_, extended.cache_key_)
            for original in (standard, extended):
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
                self.assertEqual(repeated.exact_recovery_, original.exact_recovery_)
                self.assertEqual(repeated.base_diagnostics_, original.base_diagnostics_)


if __name__ == '__main__':
    unittest.main()
