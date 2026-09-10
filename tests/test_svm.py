"""Independent SVC reconstruction and public ensemble-SVM integration tests."""

import itertools
import pickle
import tempfile
import unittest
import warnings
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold, check_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, OvOClassifier
from ensemble_dwd.svm import EnsembleKernelSVM


def binary_fixture():
    X, codes = make_classification(
        n_samples=64, n_features=5, n_informative=3, n_redundant=0,
        class_sep=.5, flip_y=.12, random_state=71)
    return X.astype(np.float64), np.where(codes == 0, 'ant', 'zebra')


def fixed_subsets(y, M=3):
    """Explicit class-balanced subsets, independent of the package sampler."""
    negative, positive = [np.flatnonzero(y == label) for label in np.unique(y)]
    subsets = []
    for seed in (11, 19, 31)[:M]:
        rng = np.random.RandomState(seed)
        chosen = np.concatenate((rng.permutation(negative)[:12],
                                 rng.permutation(positive)[:12]))
        subsets.append(rng.permutation(chosen))
    return subsets


def manual_svc(X, y, subsets, *, C, gamma, ratio, tol=1e-5):
    """Reconstruct the mathematical pipeline using only public sklearn SVC."""
    parameters = dict(C=C, gamma=gamma, kernel='rbf', tol=tol,
                      shrinking=True, max_iter=-1, cache_size=200)
    bases = [SVC(**parameters).fit(X[indices], y[indices]) for indices in subsets]
    scores = sum((base.decision_function(X) for base in bases), np.zeros(len(X))) / len(bases)
    signed_y = np.where(y == np.unique(y)[1], 1., -1.)
    distances = abs(signed_y * scores - 1.)
    # Python's explicit (distance, row index) ordering is an independent tie oracle.
    selected = np.array(sorted(range(len(X)), key=lambda i: (distances[i], i))[:int(ratio * len(X))])
    final = SVC(**parameters).fit(X[selected], y[selected])
    return bases, scores, distances, selected, final


class SVMMathematicalTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.addCleanup(self.threads.restore_original_limits)
        self.X, self.y = binary_fixture()

    def model(self, **overrides):
        parameters = dict(M=3, theta=24 / 64, sv_selection_ratio=.75,
                          C=.7, gamma=.15, tol=1e-5, random_state=42,
                          prediction_batch_size=7)
        parameters.update(overrides)
        return EnsembleKernelSVM(**parameters)

    def test_manual_string_label_reconstruction_and_fresh_final_svc(self):
        subsets = fixed_subsets(self.y)
        bases, scores, distances, selected, final = manual_svc(
            self.X, self.y, subsets, C=.7, gamma=.15, ratio=.75)
        with patch('ensemble_dwd.svm._sample_subsets', return_value=subsets):
            model = self.model().fit(self.X, self.y)
        assert_array_equal(model.classes_, ['ant', 'zebra'])
        assert_allclose(model.consensus_scores_, scores, rtol=1e-13, atol=1e-13)
        assert_allclose(model.selection_distances_, distances, rtol=1e-13, atol=1e-13)
        assert_array_equal(model.selected_indices_, selected)
        self.assertEqual(len(model.estimators_), 3)
        for actual, expected, indices in zip(model.estimators_, bases, subsets):
            self.assertIs(type(actual), SVC)
            self.assertEqual(actual.C, .7)
            self.assertEqual(actual.gamma, .15)
            assert_allclose(actual.dual_coef_, expected.dual_coef_, rtol=0, atol=0)
            assert_array_equal(actual.support_vectors_, self.X[indices][expected.support_])
        self.assertIs(type(model.final_estimator_), SVC)
        self.assertFalse(any(model.final_estimator_ is base for base in model.estimators_))
        self.assertEqual(model.final_estimator_.C, .7)
        self.assertEqual(model.final_estimator_.gamma, .15)
        query = self.X[::3] + np.array([.03, -.05, .1, -.02, .04])
        assert_array_equal(model.predict(query), final.predict(query))
        assert_allclose(model.decision_function(query), final.decision_function(query), rtol=1e-13, atol=1e-13)
        assert_allclose(model.final_estimator_.intercept_, final.intercept_, rtol=0, atol=0)
        assert_array_equal(model.n_support_, np.array([base.n_support_ for base in [*bases, final]]))
        assert_array_equal(model.support_counts_, model.n_support_.sum(axis=1))

    def test_signed_selector_distinguishes_misclassified_rows_from_unsigned_rule(self):
        subsets = fixed_subsets(self.y)
        _, scores, _, selected, _ = manual_svc(self.X, self.y, subsets,
                                              C=.7, gamma=.15, ratio=.75)
        unsigned = np.array(sorted(range(len(self.X)),
                                   key=lambda i: (abs(abs(scores[i]) - 1), i))[:48])
        self.assertFalse(np.array_equal(selected, unsigned), 'Fixture must distinguish the two rules.')
        self.assertGreater(np.count_nonzero(np.where(self.y == 'zebra', 1, -1) * scores < 0), 0)
        with patch('ensemble_dwd.svm._sample_subsets', return_value=subsets):
            model = self.model().fit(self.X, self.y)
        assert_array_equal(model.selected_indices_, selected)
        self.assertFalse(np.array_equal(model.selected_indices_, unsigned))

    def test_one_base_preserves_one_score_per_row_and_final_refit(self):
        subsets = fixed_subsets(self.y, M=1)
        _, scores, _, selected, final = manual_svc(self.X, self.y, subsets,
                                                 C=.7, gamma=.15, ratio=.75)
        with patch('ensemble_dwd.svm._sample_subsets', return_value=subsets):
            model = self.model(M=1).fit(self.X, self.y)
        self.assertEqual(model.consensus_scores_.shape, (64,))
        assert_allclose(model.consensus_scores_, scores, rtol=1e-13, atol=1e-13)
        assert_array_equal(model.selected_indices_, selected)
        assert_array_equal(model.predict(self.X), final.predict(self.X))

    def test_exact_ties_keep_input_order_and_floor_selection_size(self):
        X = np.zeros((20, 3))
        y = np.tile(['ant', 'zebra'], 10)
        with patch('ensemble_dwd.svm._sample_subsets', return_value=[np.arange(8)]):
            model = self.model(M=1, theta=.4, sv_selection_ratio=.29, gamma='scale', C=1.).fit(X, y)
        self.assertEqual(model.gamma_, 1.)
        assert_array_equal(model.consensus_scores_, np.zeros(20))
        assert_array_equal(model.selection_distances_, np.ones(20))
        assert_array_equal(model.selected_indices_, np.arange(5))

    def test_scale_gamma_is_resolved_on_supplied_fold_once_for_every_fit(self):
        subsets = fixed_subsets(self.y)
        expected = 1. / (self.X.shape[1] * self.X.var())
        self.assertFalse(np.isclose(expected, 1. / (self.X.shape[1] * self.X[subsets[0]].var())))
        with patch('ensemble_dwd.svm._sample_subsets', return_value=subsets):
            model = self.model(gamma='scale').fit(self.X, self.y)
        self.assertEqual(model.gamma_, expected)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner._gamma, expected)
            self.assertEqual(learner.C, model.C)

    def test_prediction_uses_final_svc_predict_including_its_tie_policy(self):
        model = self.model(M=1, theta=1, sv_selection_ratio=1).fit(self.X, self.y)
        sentinel = np.repeat('zebra', 3)
        with patch.object(model.final_estimator_, 'predict', return_value=sentinel) as predict:
            with patch.object(model.final_estimator_, 'decision_function', side_effect=AssertionError('Do not replace SVC.predict by thresholding.')):
                assert_array_equal(model.predict(self.X[:3]), sentinel)
        predict.assert_called_once()

    def test_batch_sizes_preserve_results_and_lean_storage_retains_final_model(self):
        full = self.model(prediction_batch_size=None).fit(self.X, self.y)
        lean = self.model(prediction_batch_size=1, store_base_estimators=False).fit(self.X, self.y)
        self.assertIsNone(lean.estimators_)
        assert_array_equal(lean.selected_indices_, full.selected_indices_)
        assert_allclose(lean.consensus_scores_, full.consensus_scores_, rtol=1e-13, atol=1e-13)
        assert_allclose(lean.decision_function(self.X), full.decision_function(self.X), rtol=1e-13, atol=1e-13)
        assert_array_equal(lean.predict(self.X), full.predict(self.X))

    def test_invalid_batch_size_after_fit_is_rejected_before_svc_and_preserves_model(self):
        model = self.model(M=1, theta=1, sv_selection_ratio=1).fit(self.X, self.y)
        query = self.X[::3]
        expected_labels = model.predict(query)
        expected_scores = model.decision_function(query)
        final_model = model.final_estimator_
        selected = model.selected_indices_.copy()
        invalid = (-1, 0, True, False, np.bool_(True), np.bool_(False),
                   1.0, -1.0, .5, np.nan, np.inf, -np.inf, '2',
                   np.float64(2.0), np.float64(np.nan))
        for value in invalid:
            for method in ('predict', 'decision_function'):
                with self.subTest(value=repr(value), value_type=type(value).__name__, method=method):
                    model.set_params(prediction_batch_size=value)
                    with patch.object(final_model, 'predict', wraps=final_model.predict) as predict, \
                         patch.object(final_model, 'decision_function', wraps=final_model.decision_function) as score:
                        with self.assertRaisesRegex(ValueError, 'prediction_batch_size'):
                            getattr(model, method)(query)
                        predict.assert_not_called()
                        score.assert_not_called()
                    self.assertIs(model.final_estimator_, final_model)
                    check_is_fitted(model)
                    assert_array_equal(model.selected_indices_, selected)
                    model.set_params(prediction_batch_size=7)
                    assert_array_equal(model.predict(query), expected_labels)
                    assert_array_equal(model.decision_function(query), expected_scores)

    def test_valid_batch_size_changes_after_fit_preserve_predictions_scores_and_learned_state(self):
        model = self.model(M=1, theta=1, sv_selection_ratio=1).fit(self.X, self.y)
        query = self.X[::3] + np.array([.03, -.05, .1, -.02, .04])
        expected_labels = model.final_estimator_.predict(query)
        expected_scores = model.final_estimator_.decision_function(query)
        final_model = model.final_estimator_
        bases = tuple(model.estimators_)
        learned = {name: getattr(model, name).copy() for name in
                   ('classes_', 'selected_indices_', 'consensus_scores_', 'selection_distances_',
                    'n_iter_', 'fit_status_', 'converged_', 'n_support_', 'support_counts_')}
        subsets = [indices.copy() for indices in model.subset_indices_]
        for size in (None, 1, np.int64(3), len(query) + 17):
            with self.subTest(size=size):
                model.set_params(prediction_batch_size=size)
                assert_array_equal(model.predict(query), expected_labels)
                assert_array_equal(model.decision_function(query), expected_scores)
                self.assertIs(model.final_estimator_, final_model)
                self.assertTrue(all(actual is old for actual, old in zip(model.estimators_, bases)))
                for name, expected in learned.items():
                    assert_array_equal(getattr(model, name), expected)
                for actual, expected in zip(model.subset_indices_, subsets):
                    assert_array_equal(actual, expected)
                check_is_fitted(model)

    def test_pipeline_batch_size_updates_validate_before_svc_and_allow_recovery(self):
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('svm', self.model(M=1, theta=1, sv_selection_ratio=1))]).fit(self.X, self.y)
        query = self.X[::3]
        expected_labels = pipeline.predict(query)
        expected_scores = pipeline.decision_function(query)
        final_model = pipeline.named_steps['svm'].final_estimator_
        scaler_mean = pipeline.named_steps['scale'].mean_.copy()
        pipeline.set_params(svm__prediction_batch_size=-1)
        for method in ('predict', 'decision_function'):
            with self.subTest(method=method):
                with patch.object(final_model, 'predict', wraps=final_model.predict) as predict, \
                     patch.object(final_model, 'decision_function', wraps=final_model.decision_function) as score:
                    with self.assertRaisesRegex(ValueError, 'prediction_batch_size'):
                        getattr(pipeline, method)(query)
                    predict.assert_not_called()
                    score.assert_not_called()
        pipeline.set_params(svm__prediction_batch_size=np.int64(1))
        assert_array_equal(pipeline.predict(query), expected_labels)
        assert_array_equal(pipeline.decision_function(query), expected_scores)
        self.assertIs(pipeline.named_steps['svm'].final_estimator_, final_model)
        assert_array_equal(pipeline.named_steps['scale'].mean_, scaler_mean)

    def test_same_seed_repeats_without_mutating_the_global_random_stream(self):
        before = np.random.get_state()
        first = self.model().fit(self.X, self.y)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        second = clone(first).fit(self.X, self.y)
        for left, right in zip(first.subset_indices_, second.subset_indices_):
            assert_array_equal(left, right)
            self.assertEqual(len(np.unique(left)), 24)
            self.assertEqual(len(np.unique(self.y[left])), 2)
        assert_array_equal(first.predict(self.X), second.predict(self.X))

    def test_max_iteration_exhaustion_is_reported_without_silent_solver_substitution(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = self.model(max_iter=1, C=100., gamma=.5, sv_selection_ratio=1).fit(self.X, self.y)
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))
        assert_array_equal(model.fit_status_, np.ones(4, dtype=int))
        assert_array_equal(model.converged_, np.zeros(4, dtype=bool))
        assert_array_equal(model.n_iter_, np.ones(4, dtype=int))
        self.assertTrue(np.isfinite(model.decision_function(self.X)).all())

    def test_failed_refit_clears_previously_fitted_state(self):
        model = self.model(M=1, theta=1, sv_selection_ratio=1).fit(self.X, self.y)
        factory = model._make_estimator
        calls = []
        def injected(gamma):
            learner = factory(gamma)
            calls.append(learner)
            if len(calls) == 2:
                learner.fit = lambda *args, **kwargs: (_ for _ in ()).throw(FloatingPointError('injected final failure'))
            return learner
        with patch.object(model, '_make_estimator', side_effect=injected):
            with self.assertRaisesRegex(FloatingPointError, 'injected final failure'):
                model.fit(self.X, self.y)
        for name in ('final_estimator_', 'classes_', 'selected_indices_', 'consensus_scores_', 'n_features_in_'):
            self.assertFalse(hasattr(model, name), name)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_ineligible_selection_is_explicit_not_a_base_or_full_data_fallback(self):
        model = self.model(M=1, sv_selection_ratio=.01)
        with self.assertRaises(IneligibleSelectionError):
            model.fit(self.X, self.y)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)


class SVMCacheAndIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.addCleanup(self.threads.restore_original_limits)
        self.X, self.y = binary_fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def model(self, **overrides):
        parameters = dict(M=2, theta=.5, sv_selection_ratio=.75, C=.7, gamma=.15,
                          random_state=42, cache_dir=self.temporary.name)
        parameters.update(overrides)
        return EnsembleKernelSVM(**parameters)

    def test_rho_reuses_bases_but_has_a_fresh_final_fit_and_cache_off_parity(self):
        first = self.model().fit(self.X, self.y)
        second = self.model(sv_selection_ratio=1).fit(self.X, self.y)
        independent = self.model(sv_selection_ratio=1, cache_dir=None).fit(self.X, self.y)
        self.assertFalse(first.cache_hit_); self.assertTrue(second.cache_hit_)
        self.assertEqual(first.cache_key_, second.cache_key_)
        self.assertIsNot(first.final_estimator_, second.final_estimator_)
        self.assertEqual(len(first.selected_indices_), 48)
        self.assertEqual(len(second.selected_indices_), 64)
        assert_array_equal(first.consensus_scores_, second.consensus_scores_)
        assert_array_equal(second.selected_indices_, independent.selected_indices_)
        assert_array_equal(second.consensus_scores_, independent.consensus_scores_)
        assert_array_equal(second.predict(self.X), independent.predict(self.X))
        assert_allclose(second.decision_function(self.X), independent.decision_function(self.X), rtol=0, atol=0)

    def test_cache_separates_C_gamma_row_order_labels_and_different_folds(self):
        baseline = self.model().fit(self.X, self.y)
        cases = [({'C': 2.}, self.X, self.y), ({'gamma': .3}, self.X, self.y),
                 ({}, self.X[::-1], self.y[::-1]), ({}, self.X, np.where(self.y == 'ant', 'left', 'right')),
                 ({}, self.X[:48], self.y[:48])]
        for options, X, y in cases:
            with self.subTest(options=options, n=len(X), label=str(y[0])):
                model = self.model(**options).fit(X, y)
                self.assertFalse(model.cache_hit_)
                self.assertNotEqual(model.cache_key_, baseline.cache_key_)

    def test_cache_cannot_reuse_DWD_bases_for_SVM(self):
        svm = self.model(M=1, sv_selection_ratio=1).fit(self.X, self.y)
        dwd = EnsembleKernelDWD(M=1, theta=.5, sv_selection_ratio=1, gamma=.15,
            lambd=.1, max_iter=3, stopping='fixed', random_state=42,
            cache_dir=self.temporary.name).fit(self.X, self.y)
        self.assertFalse(dwd.cache_hit_)
        self.assertNotEqual(svm.cache_key_, dwd.cache_key_)
        svm_again = self.model(M=1, sv_selection_ratio=1).fit(self.X, self.y)
        self.assertTrue(svm_again.cache_hit_)
        self.assertIs(type(svm_again.final_estimator_), SVC)

    def test_cached_nonconvergence_remains_visible_as_warning_and_diagnostics(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            first = self.model(max_iter=1, C=100.).fit(self.X, self.y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warm = self.model(max_iter=1, C=100., sv_selection_ratio=1).fit(self.X, self.y)
        self.assertTrue(warm.cache_hit_)
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning)
                            and 'cached base' in str(item.message) for item in caught))
        assert_array_equal(warm.fit_status_[:-1], first.fit_status_[:-1])
        assert_array_equal(warm.converged_[:-1], np.zeros(2, dtype=bool))

    def test_failed_cached_final_refit_clears_model_but_preserves_valid_base_cache(self):
        model = self.model().fit(self.X, self.y)
        saved_key = model.cache_key_
        native_factory = model._make_estimator
        def injected(gamma):
            learner = native_factory(gamma)
            learner.fit = lambda *args, **kwargs: (_ for _ in ()).throw(FloatingPointError('cached final failure'))
            return learner
        with patch.object(model, '_make_estimator', side_effect=injected) as factory:
            with self.assertRaisesRegex(FloatingPointError, 'cached final failure'):
                model.fit(self.X, self.y)
        self.assertEqual(factory.call_count, 1, 'A warm fit should create only the final SVC.')
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        self.assertFalse(hasattr(model, 'classes_'))
        restored = self.model().fit(self.X, self.y)
        self.assertTrue(restored.cache_hit_)
        self.assertEqual(restored.cache_key_, saved_key)

    def test_clone_tags_validation_and_serialization(self):
        model = self.model(cache_dir=None)
        self.assertTrue(is_classifier(model))
        self.assertIsInstance(check_cv(2, self.y, classifier=is_classifier(model)), StratifiedKFold)
        self.assertEqual(clone(model).get_params(), model.get_params())
        with self.assertRaises(NotFittedError):
            model.predict(self.X)
        fitted = model.fit(self.X, self.y)
        restored = pickle.loads(pickle.dumps(fitted))
        assert_array_equal(restored.predict(self.X), fitted.predict(self.X))
        with self.assertRaises(ValueError):
            restored.predict(self.X[:, :-1])
        for options in ({'M': 0}, {'C': 0.}, {'gamma': 0.}, {'theta': 1.1},
                        {'max_iter': 0}, {'tol': 0.}, {'kernel': 'linear'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options).fit(self.X, self.y)

    def test_full_sklearn_estimator_contract_on_valid_full_selection(self):
        check_estimator(EnsembleKernelSVM(M=1, theta=1, sv_selection_ratio=1,
                                         cache_dir=None, random_state=42), on_skip=None)

    def test_pipeline_gridsearch_real_two_workers_matches_manual_fold_scaling(self):
        folds = list(StratifiedKFold(2, shuffle=True, random_state=7).split(self.X, self.y))
        pipeline = Pipeline([('scale', StandardScaler()), ('svm', self.model())])
        candidates = {'svm__C': [.3, 1.], 'svm__sv_selection_ratio': [.75, 1.]}
        search = GridSearchCV(pipeline, candidates, cv=folds, n_jobs=2,
                              pre_dispatch=2, error_score='raise')
        with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
            search.fit(self.X, self.y)
        self.assertTrue(is_classifier(search.best_estimator_))
        assert_array_equal(search.classes_, np.unique(self.y))
        for row, parameters in enumerate(search.cv_results_['params']):
            for fold, (train, validation) in enumerate(folds):
                scaler = StandardScaler().fit(self.X[train])
                expected = self.model(C=parameters['svm__C'],
                    sv_selection_ratio=parameters['svm__sv_selection_ratio'], cache_dir=None)
                expected.fit(scaler.transform(self.X[train]), self.y[train])
                accuracy = np.mean(expected.predict(scaler.transform(self.X[validation])) == self.y[validation])
                self.assertEqual(search.cv_results_[f'split{fold}_test_score'][row], accuracy)
        assert_allclose(search.best_estimator_.named_steps['scale'].mean_, self.X.mean(axis=0))

    def test_all_ten_class_ovo_has_45_pairs_and_native_svc_vote_parity(self):
        X, codes = make_blobs(n_samples=60, centers=10, n_features=3,
                             cluster_std=.6, random_state=16)
        labels = np.array([f'digit-{i}' for i in range(10)])[codes]
        base = self.model(M=1, theta=1, sv_selection_ratio=1, cache_dir=None)
        model = OvOClassifier(base, prediction_batch_size=7).fit(X, labels)
        self.assertEqual(len(model.estimators_), 45)
        expected = np.zeros((len(X), 10), dtype=int)
        for pair, child in zip(itertools.combinations(range(10), 2), model.estimators_):
            votes = child.final_estimator_.predict(X)
            destinations = np.where(votes == 1, pair[0], pair[1])
            expected[np.arange(len(X)), destinations] += 1
            self.assertIs(type(child.final_estimator_), SVC)
            self.assertEqual(child.C, base.C)
            self.assertEqual(child.gamma, base.gamma)
        assert_array_equal(expected.sum(axis=1), np.full(len(X), 45))
        assert_array_equal(model.predict(X), model.classes_[expected.argmax(axis=1)])


if __name__ == '__main__':
    unittest.main()
