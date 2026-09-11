"""Multiclass voting, estimator, and binary-ensemble integration contracts."""

import itertools
import pickle
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold, check_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, IneligibleSubsetError, OvOClassifier


class RecordingBinary(ClassifierMixin, BaseEstimator):
    """A small deterministic classifier with observable pair training data."""

    def __init__(self, offset=0.0, prediction="nearest", fail_pair=None,
                 failure="numerical"):
        self.offset = offset
        self.prediction = prediction
        self.fail_pair = fail_pair
        self.failure = failure

    def fit(self, X, y):
        self.X_ = np.array(X, copy=True)
        self.y_ = np.array(y, copy=True)
        self.classes_ = np.unique(y)
        self.n_features_in_ = self.X_.shape[1]
        self.batch_lengths_ = []
        self.coordinates_ = np.unique(self.X_[:, 0])
        if (self.fail_pair is not None and
                tuple(self.coordinates_) == tuple(self.fail_pair)):
            if self.failure == "ineligible":
                raise IneligibleSelectionError("injected ineligible selection")
            if self.failure == "subset":
                raise IneligibleSubsetError("injected single-class base subset")
            raise FloatingPointError("injected numerical failure")
        self.centers_ = np.array([
            self.X_[self.y_ == label, 0].mean() for label in self.classes_
        ])
        return self

    def predict(self, X):
        self.batch_lengths_.append(len(X))
        if self.prediction == "negative":
            return np.full(len(X), -1, dtype=int)
        if self.prediction == "cycle":
            # 0 beats 1, 2 beats 0, and 1 beats 2: all receive one vote.
            vote = -1 if tuple(self.coordinates_) == (0.0, 2.0) else 1
            return np.full(len(X), vote, dtype=int)
        if self.prediction == "bad_label":
            return np.full(len(X), 7)
        if self.prediction == "bad_shape":
            return np.ones((len(X), 1))
        distances = abs(np.asarray(X)[:, 0, None] - self.centers_[None, :])
        distances[:, 1] += self.offset
        return self.classes_[np.argmin(distances, axis=1)]


class RecordingScaler(StandardScaler):
    fitted_inputs = []

    def fit(self, X, y=None, sample_weight=None):
        type(self).fitted_inputs.append(np.array(X, copy=True))
        return super().fit(X, y, sample_weight=sample_weight)


class ZeroMarginDWD:
    """A DWD stand-in exercising zero margins and cached ensemble stages."""

    fits = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.X_ = np.array(X, copy=True)
        self.y_ = np.array(y, copy=True)
        self.classes_ = np.unique(y)
        self.n_iter_ = 1
        self.converged_ = True
        type(self).fits.append(self)
        return self

    def decision_function(self, X):
        return np.zeros(len(X))


def pair_data(n_classes=3, samples_per_class=8, labels=None):
    codes = np.repeat(np.arange(n_classes), samples_per_class)
    X = np.column_stack((codes, np.arange(len(codes))))
    values = np.arange(n_classes) if labels is None else np.asarray(labels)
    return X, values[codes]


class MulticlassContractTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = pair_data()

    def estimator(self, **kwargs):
        return OvOClassifier(RecordingBinary(), **kwargs)

    def test_all_ten_classes_make_45_isolated_cloned_binary_pairs(self):
        X, y = pair_data(n_classes=10, samples_per_class=4)
        original_X, original_y = X.copy(), y.copy()
        model = self.estimator().fit(X, y)
        expected_pairs = list(itertools.combinations(np.unique(y), 2))
        self.assertEqual(list(model.pairs_), expected_pairs)
        self.assertEqual(len(model.estimators_), 45)
        self.assertEqual(len({id(item) for item in model.estimators_}), 45)
        self.assertFalse(hasattr(model.estimator, "classes_"))
        for pair, indices, child in zip(model.pairs_, model.pair_indices_,
                                         model.estimators_):
            expected_indices = np.flatnonzero((y == pair[0]) | (y == pair[1]))
            assert_array_equal(indices, expected_indices)
            assert_array_equal(child.X_, X[expected_indices])
            assert_array_equal(child.y_, np.where(y[expected_indices] == pair[0], 1, -1))
            self.assertEqual(child.X_.dtype, np.float64)
            assert_array_equal(child.classes_, [-1, 1])
        assert_array_equal(model.predict(X), y)
        assert_array_equal(X, original_X)
        assert_array_equal(y, original_y)

    def test_vote_counts_match_independent_pair_prediction_tally(self):
        model = self.estimator(prediction_batch_size=5).fit(self.X, self.y)
        X = np.column_stack((np.linspace(-0.5, 2.5, 17), np.zeros(17)))
        expected = np.zeros((len(X), 3), dtype=int)
        for pair, child in zip(model.pairs_, model.estimators_):
            pair_predictions = child.predict(X)
            for row, predicted in enumerate(pair_predictions):
                label = pair[0] if predicted == 1 else pair[1]
                expected[row, np.flatnonzero(model.classes_ == label)[0]] += 1
        votes = model.vote_counts(X)
        self.assertTrue(np.issubdtype(votes.dtype, np.integer))
        assert_array_equal(votes, expected)
        assert_array_equal(votes.sum(axis=1), 3)
        assert_array_equal(model.predict(X), model.classes_[expected.argmax(axis=1)])

    def test_cyclic_votes_use_smallest_original_label_without_confidence_tiebreak(self):
        for labels in [[10, 30, 99], ["ant", "bee", "cat"]]:
            with self.subTest(labels=labels):
                X, y = pair_data(labels=labels)
                model = OvOClassifier(RecordingBinary(prediction="cycle")).fit(X, y)
                assert_array_equal(model.vote_counts(X[:4]), np.ones((4, 3), dtype=int))
                assert_array_equal(model.predict(X[:4]), np.repeat(labels[0], 4))

    def test_string_boolean_and_large_integer_label_dtype_is_preserved(self):
        fixtures = [["aardvark", "bear", "cat"],
                    np.array([2**60 + 1, 2**60 + 3, 2**60 + 7], dtype=np.int64),
                    np.array([False, True], dtype=bool)]
        for labels in fixtures:
            with self.subTest(labels=labels):
                X, y = pair_data(len(labels), labels=labels)
                model = self.estimator().fit(X, y)
                predicted = model.predict(X)
                assert_array_equal(predicted, y)
                self.assertEqual(predicted.dtype, y.dtype)
                self.assertEqual(model.classes_.dtype, y.dtype)

    def test_binary_wrapper_preserves_original_script_label_orientation(self):
        X, y = pair_data(2, labels=["first", "second"])
        model = OvOClassifier(RecordingBinary(prediction="negative")).fit(X, y)
        assert_array_equal(model.estimators_[0].y_, np.where(y == "first", 1, -1))
        assert_array_equal(model.predict(X), np.repeat("second", len(y)))
        assert_array_equal(model.vote_counts(X), np.tile([0, 1], (len(y), 1)))

    def test_prediction_batches_do_not_change_votes_and_bound_child_calls(self):
        small = self.estimator(prediction_batch_size=5).fit(self.X, self.y)
        large = self.estimator(prediction_batch_size=1000).fit(self.X, self.y)
        assert_array_equal(small.predict(self.X), large.predict(self.X))
        assert_array_equal(small.vote_counts(self.X), large.vote_counts(self.X))
        for child in small.estimators_:
            self.assertLessEqual(max(child.batch_lengths_), 5)
            self.assertEqual(sum(child.batch_lengths_), 2 * len(self.X))
        assert_array_equal(small.predict(self.X[:1]), self.y[:1])

    def test_classifier_tags_clone_and_nested_parameters_support_stratified_cv(self):
        model = self.estimator()
        self.assertTrue(is_classifier(model))
        self.assertTrue(model.__sklearn_tags__().classifier_tags.multi_class)
        self.assertIsInstance(check_cv(3, self.y, classifier=is_classifier(model)),
                              StratifiedKFold)
        copied = clone(model)
        self.assertIsNot(copied.estimator, model.estimator)
        self.assertIs(copied.set_params(estimator__offset=0.25), copied)
        self.assertEqual(copied.estimator.offset, 0.25)
        self.assertEqual(model.estimator.offset, 0.0)
        self.assertFalse(hasattr(copied, "estimators_"))

    def test_public_multiclass_estimator_passes_sklearn_contract_checks(self):
        check_estimator(OvOClassifier(SVC()), on_skip=None)

    def test_gridsearch_fits_preprocessing_only_on_each_training_fold(self):
        X, y = pair_data(samples_per_class=12)
        pipeline = Pipeline([
            ("scale", RecordingScaler()),
            ("ovo", self.estimator()),
        ])
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=31).split(X, y))
        RecordingScaler.fitted_inputs = []
        search = GridSearchCV(pipeline, {"ovo__estimator__offset": [0.0, 0.2]},
                              cv=folds, scoring="accuracy", n_jobs=1,
                              error_score="raise").fit(X, y)
        self.assertEqual(len(RecordingScaler.fitted_inputs), 7)
        for candidate in range(2):
            for fold, (train, _) in enumerate(folds):
                observed = RecordingScaler.fitted_inputs[candidate * 3 + fold]
                assert_array_equal(observed, X[train])
                self.assertLess(len(observed), len(X))
        assert_array_equal(RecordingScaler.fitted_inputs[-1], X)
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
        self.assertEqual(search.best_estimator_["ovo"].estimator.offset,
                         search.best_params_["ovo__estimator__offset"])
        for child in search.best_estimator_["ovo"].estimators_:
            self.assertEqual(child.offset, search.best_estimator_["ovo"].estimator.offset)
        assert_allclose(search.best_estimator_["scale"].mean_, X.mean(axis=0))

    def test_parallel_pair_training_matches_serial_and_pickle_roundtrip(self):
        serial = self.estimator(n_jobs=1).fit(self.X, self.y)
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            parallel = self.estimator(n_jobs=2).fit(self.X, self.y)
        self.assertEqual(parallel.pairs_, serial.pairs_)
        for a, b in zip(parallel.estimators_, serial.estimators_):
            assert_array_equal(a.X_, b.X_)
            assert_array_equal(a.y_, b.y_)
        assert_array_equal(parallel.predict(self.X), serial.predict(self.X))
        assert_array_equal(parallel.vote_counts(self.X), serial.vote_counts(self.X))
        restored = pickle.loads(pickle.dumps(parallel))
        assert_array_equal(restored.predict(self.X), serial.predict(self.X))

    def test_numerical_pair_failure_is_visible_and_clears_previous_and_partial_state(self):
        model = self.estimator().fit(self.X, self.y)
        model.set_params(estimator__fail_pair=(0.0, 2.0))
        with self.assertRaisesRegex(FloatingPointError, "injected numerical failure") as caught:
            model.fit(self.X, self.y)
        self.assertTrue(any("OVO pair" in note for note in caught.exception.__notes__))
        self.assertFalse(any(name.endswith("_") for name in vars(model)))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_ineligible_pair_propagates_without_fallback_and_can_refit_after_correction(self):
        model = OvOClassifier(RecordingBinary(fail_pair=(0.0, 2.0), failure="ineligible"))
        with self.assertRaisesRegex(IneligibleSelectionError, "OVO pair"):
            model.fit(self.X, self.y)
        with self.assertRaises(NotFittedError):
            model.vote_counts(self.X)
        self.assertFalse(any(name.endswith("_") for name in vars(model)))
        model.set_params(estimator__fail_pair=None).fit(self.X, self.y)
        assert_array_equal(model.predict(self.X), self.y)

    def test_ineligible_subset_preserves_type_pair_context_and_clears_state(self):
        for jobs in (1, 2):
            with self.subTest(n_jobs=jobs):
                model = OvOClassifier(RecordingBinary(), n_jobs=jobs).fit(self.X, self.y)
                model.set_params(estimator__fail_pair=(0.0, 2.0), estimator__failure='subset')
                with self.assertRaisesRegex(IneligibleSubsetError, 'OVO pair') as caught:
                    model.fit(self.X, self.y)
                self.assertIsInstance(caught.exception, IneligibleSelectionError)
                self.assertIn('single-class base subset', str(caught.exception))
                self.assertFalse(any(name.endswith('_') for name in vars(model)))
                with self.assertRaises(NotFittedError):
                    model.predict(self.X)

    def test_invalid_prediction_labels_and_shapes_raise_instead_of_becoming_votes(self):
        for prediction in ["bad_label", "bad_shape"]:
            with self.subTest(prediction=prediction):
                model = OvOClassifier(RecordingBinary(prediction=prediction)).fit(self.X, self.y)
                with self.assertRaises((ValueError, TypeError)):
                    model.predict(self.X)

    def test_invalid_options_and_estimators_fail_before_leaving_fitted_state(self):
        invalid = {"n_jobs": [0, True, 1.5, "2"],
                   "prediction_batch_size": [0, -1, True, 1.5, None]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(parameter=key, value=value):
                    model = self.estimator(**{key: value})
                    with self.assertRaises((ValueError, TypeError)):
                        model.fit(self.X, self.y)
                    with self.assertRaises(NotFittedError):
                        check_is_fitted(model)
        for estimator in [None, object()]:
            with self.subTest(estimator=estimator):
                with self.assertRaises((ValueError, TypeError)):
                    OvOClassifier(estimator).fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "precomputed"):
            OvOClassifier(SVC(kernel="precomputed")).fit(self.X @ self.X.T, self.y)
        self.estimator(n_jobs=None).fit(self.X, self.y)

    def test_dense_finite_classification_input_and_matching_prediction_features(self):
        bad_X = [sparse.csr_matrix(self.X), self.X[:, 0],
                 np.full_like(self.X, np.nan, dtype=float),
                 np.full_like(self.X, np.inf, dtype=float)]
        for X in bad_X:
            with self.subTest(kind=type(X), shape=np.shape(X)):
                with self.assertRaises((TypeError, ValueError)):
                    self.estimator().fit(X, self.y)
        for y in [np.zeros(len(self.y)), self.y[:-1], np.linspace(0.01, 0.99, len(self.y))]:
            with self.subTest(y=y):
                with self.assertRaises(ValueError):
                    self.estimator().fit(self.X, y)
        model = self.estimator().fit(self.X, self.y)
        for X in [self.X[:, :1], self.X[:, 0], sparse.csr_matrix(self.X),
                  np.full_like(self.X, np.nan, dtype=float)]:
            with self.subTest(prediction_shape=np.shape(X)):
                with self.assertRaises((TypeError, ValueError)):
                    model.predict(X)

    def test_feature_names_are_preserved_and_checked_when_pandas_is_available(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional")
        X = pd.DataFrame(self.X, columns=["position", "row_id"])
        model = self.estimator().fit(X, self.y)
        assert_array_equal(model.feature_names_in_, X.columns)
        assert_array_equal(model.predict(X), self.y)
        with self.assertRaises(ValueError):
            model.predict(X[["row_id", "position"]])


class MulticlassBinaryEnsembleTests(unittest.TestCase):
    def setUp(self):
        ZeroMarginDWD.fits = []
        self.X, self.y = pair_data(labels=["a", "b", "c"])

    def test_m_one_zero_margin_and_shared_optimized_dwd_parameters(self):
        base = EnsembleKernelDWD(M=1, theta=0.5, sv_selection_ratio=1.0,
                                 gamma=0.01, lambd=0.003, q=0.5,
                                 stopping="fixed", max_iter=7, random_state=17)
        with patch("ensemble_dwd.estimator.KernGDWD", ZeroMarginDWD):
            model = OvOClassifier(base).fit(self.X, self.y)
        self.assertEqual(len(ZeroMarginDWD.fits), 3 * (base.M + 1))
        # A zero DWD margin predicts -1, which is the second original pair label.
        assert_array_equal(model.vote_counts(self.X), np.tile([0, 1, 2], (len(self.y), 1)))
        assert_array_equal(model.predict(self.X), np.repeat("c", len(self.y)))
        for child in model.estimators_:
            self.assertEqual(len(child.estimators_), 1)
            self.assertEqual(child.consensus_scores_.shape, (16,))
            for learner in [*child.estimators_, child.final_estimator_]:
                self.assertEqual(learner.params["implementation"], "optimized")
                self.assertEqual(learner.params["kernel_kws"]["gamma"], base.gamma)
                for parameter in ["lambd", "q", "stopping", "max_iter"]:
                    self.assertEqual(learner.params[parameter], getattr(base, parameter))

    def test_selection_ratio_change_reuses_each_pairs_cached_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            base = EnsembleKernelDWD(M=2, theta=0.5, sv_selection_ratio=1.0,
                                     gamma=0.01, random_state=17, cache_dir=directory)
            with patch("ensemble_dwd.estimator.KernGDWD", ZeroMarginDWD):
                first = OvOClassifier(base).fit(self.X, self.y)
                fits_after_first = len(ZeroMarginDWD.fits)
                second = clone(first).set_params(estimator__sv_selection_ratio=0.75)
                second.fit(self.X, self.y)
            self.assertEqual(fits_after_first, 9)
            self.assertEqual(len(ZeroMarginDWD.fits), fits_after_first + 3)
            for before, after in zip(first.estimators_, second.estimators_):
                self.assertFalse(before.cache_hit_)
                self.assertTrue(after.cache_hit_)
                self.assertEqual(before.cache_key_, after.cache_key_)
                assert_array_equal(before.consensus_scores_, after.consensus_scores_)
                self.assertEqual(len(before.selected_indices_), 16)
                self.assertEqual(len(after.selected_indices_), 12)

    def test_scale_gamma_is_resolved_per_pair_and_then_shared_by_its_learners(self):
        base = EnsembleKernelDWD(M=2, theta=0.5, sv_selection_ratio=1.0,
                                 gamma="scale", random_state=17)
        with patch("ensemble_dwd.estimator.KernGDWD", ZeroMarginDWD):
            model = OvOClassifier(base).fit(self.X, self.y)
        resolved = []
        for child, rows in zip(model.estimators_, model.pair_indices_):
            expected = 1.0 / (self.X.shape[1] * self.X[rows].var())
            self.assertEqual(child.gamma_, expected)
            resolved.append(child.gamma_)
            for learner in [*child.estimators_, child.final_estimator_]:
                self.assertEqual(learner.params["kernel_kws"]["gamma"], expected)
        self.assertGreater(np.ptp(resolved), 0.0)


if __name__ == "__main__":
    unittest.main()
