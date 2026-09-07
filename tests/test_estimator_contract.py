"""Algorithm and estimator contracts using a deterministic DWD stand-in."""

import itertools
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd.estimator import _sample_subsets


class RecordingDWD:
    """A fitted affine margin whose scale depends on the sampled training set."""

    instances = []
    fail_at = None

    def __init__(self, **params):
        self.params = params
        self.index = len(type(self).instances)
        type(self).instances.append(self)

    def fit(self, X, y):
        if self.index == type(self).fail_at:
            raise FloatingPointError("injected numerical failure")
        self.X = np.array(X, copy=True)
        self.y = np.array(y, copy=True)
        self.classes_ = np.unique(y)
        self.n_iter_ = self.index + 2
        self.converged_ = self.index % 2 == 0
        self.scale = 0.4 + np.mean(X[:, 0] ** 2)
        self.offset = 0.2 * np.mean(X[:, 1])
        return self

    def decision_function(self, X):
        self.last_prediction_X = np.asarray(X)
        return self.scale * np.asarray(X)[:, 0] + self.offset

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class ConstantDWD(RecordingDWD):
    def decision_function(self, X):
        return np.zeros(len(X))


class ContractTests(unittest.TestCase):
    def setUp(self):
        RecordingDWD.instances = []
        RecordingDWD.fail_at = None
        ConstantDWD.instances = []
        ConstantDWD.fail_at = None
        self.X = np.column_stack((np.linspace(-2, 2, 20),
                                  np.sin(np.arange(20))))
        self.y = np.tile(["left", "right"], 10)

    def model(self, **kwargs):
        params = dict(M=3, theta=0.4, sv_selection_ratio=0.5,
                      gamma=0.02, random_state=17)
        params.update(kwargs)
        return EnsembleKernelDWD(**params)

    def fit_spy(self, model=None, X=None, y=None, spy=RecordingDWD):
        if model is None:
            model = self.model()
        with patch("ensemble_dwd.estimator.KernGDWD", spy):
            model.fit(self.X if X is None else X, self.y if y is None else y)
        return model

    def test_selection_is_absolute_mean_raw_margin_and_fresh_final_fit(self):
        model = self.fit_spy()
        # Independent affine-margin identity for the fitted stand-ins.
        slopes = [0.4 + sum(self.X[i, 0] ** 2 for i in subset) / len(subset)
                  for subset in model.subset_indices_]
        offsets = [0.2 * sum(self.X[i, 1] for i in subset) / len(subset)
                   for subset in model.subset_indices_]
        expected_scores = (sum(slopes) * self.X[:, 0] + sum(offsets)) / model.M
        expected_order = sorted(range(len(self.X)),
                                key=lambda i: (abs(expected_scores[i]), i))
        expected_indices = expected_order[:10]
        assert_allclose(model.consensus_scores_, expected_scores, rtol=1e-14, atol=1e-14)
        assert_array_equal(model.selected_indices_, expected_indices)
        self.assertEqual(len(RecordingDWD.instances), model.M + 1)
        self.assertIs(model.final_estimator_, RecordingDWD.instances[-1])
        self.assertNotIn(model.final_estimator_, model.estimators_)
        assert_array_equal(model.final_estimator_.X, self.X[expected_indices])
        self.assertEqual(len(np.unique(model.final_estimator_.y)), 2)
        assert_array_equal(model.n_iter_, [2, 3, 4, 5])
        assert_array_equal(model.converged_, [True, False, True, False])

    def test_tied_scores_select_original_order_exact_floor(self):
        model = self.fit_spy(self.model(sv_selection_ratio=0.29), spy=ConstantDWD)
        assert_array_equal(model.selected_indices_, np.arange(5))
        self.assertEqual(len(model.final_estimator_.X), 5)

    def test_raw_margins_are_not_votes_or_normalized_scores(self):
        margins = np.array([[100, 0.1, 3, 4, -6, -5, -4, -3],
                            [-98, 0.1, -2, -3, 7, 8, -4, -3]], dtype=float)

        class ScriptedDWD(RecordingDWD):
            def decision_function(self, X):
                if self.index < 2:
                    return margins[self.index, np.asarray(X)[:, 0].astype(int)]
                return super().decision_function(X)

        X = np.column_stack([np.arange(8), np.zeros(8)])
        y = np.tile(["left", "right"], 4)
        model = self.fit_spy(self.model(M=2, theta=0.5), X=X, y=y, spy=ScriptedDWD)
        assert_array_equal(model.selected_indices_, [1, 2, 3, 4])
        assert_allclose(model.consensus_scores_, [1, 0.1, 0.5, 0.5, 0.5, 1.5, -4, -3])
        voting = np.argsort(abs(np.sign(margins).mean(axis=0)), kind="stable")[:4]
        normalized = margins / np.linalg.norm(margins, axis=1)[:, None]
        normalized_order = np.argsort(abs(normalized.mean(axis=0)), kind="stable")[:4]
        self.assertFalse(np.array_equal(model.selected_indices_, voting))
        self.assertFalse(np.array_equal(model.selected_indices_, normalized_order))

    def test_minimum_subset_size_and_m_one_shapes(self):
        model = self.fit_spy(self.model(M=1, theta=0.001, sv_selection_ratio=1.0))
        self.assertEqual(len(model.subset_indices_), 1)
        self.assertEqual(len(model.subset_indices_[0]), 2)
        self.assertEqual(np.shape(model.consensus_scores_), (20,))
        self.assertEqual(np.shape(model.n_iter_), (2,))
        self.assertEqual(np.shape(model.converged_), (2,))
        self.assertEqual(np.shape(model.decision_function(self.X[:1])), (1,))

    def test_shared_fit_parameters_and_forced_optimized_solver(self):
        model = self.fit_spy(self.model(lambd=0.013, q=0.5, max_iter=11,
                                       obj_tol=2e-7, stopping="fixed", tol=3e-7,
                                       acceleration="restart", prediction_batch_size=7))
        for learner in RecordingDWD.instances:
            params = learner.params
            self.assertEqual(params["implementation"], "optimized")
            self.assertEqual(params["solver_mode"], "schur")
            self.assertEqual(params["backend"], "auto")
            self.assertEqual(params["initialization"], "zero")
            for key in ["lambd", "q", "kernel", "max_iter", "obj_tol",
                        "stopping", "tol", "acceleration", "prediction_batch_size"]:
                self.assertEqual(params[key], getattr(model, key), key)
            self.assertEqual(params["kernel_kws"]["gamma"], model.gamma_)

    def test_scale_gamma_resolved_on_entire_current_fold(self):
        model = self.fit_spy(self.model(gamma="scale"))
        expected = 1.0 / (self.X.shape[1] * self.X.var())
        self.assertEqual(model.gamma_, expected)
        self.assertTrue(all(learner.params["kernel_kws"]["gamma"] == expected
                            for learner in RecordingDWD.instances))
        self.assertNotEqual(expected, 1.0 / (2 * RecordingDWD.instances[0].X.var()))

    def test_auto_and_constant_scale_gamma(self):
        for gamma, X, expected in [("auto", self.X, 0.5),
                                   ("scale", np.ones_like(self.X), 1.0)]:
            with self.subTest(gamma=gamma, constant=bool(np.ptp(X) == 0)):
                model = self.fit_spy(self.model(gamma=gamma, sv_selection_ratio=1.0), X=X)
                self.assertEqual(model.gamma_, expected)

    def test_scale_gamma_avoids_intermediate_reciprocal_overflow(self):
        X = np.array([[-7e-155] * 4, [7e-155] * 4], dtype=float)
        gamma, keywords = self.model(gamma="scale")._kernel_parameters(X)
        self.assertTrue(np.isfinite(gamma))
        self.assertGreater(gamma, 1e307)
        self.assertEqual(gamma, (1.0 / X.shape[1]) / X.var())
        self.assertEqual(keywords["gamma"], gamma)

    def test_unused_gamma_does_not_compute_overflowing_variance(self):
        X = np.array([[-1e200, -1e200], [1e200, 1e200]])
        for kernel in ["linear", lambda A, B: A @ B.T]:
            with self.subTest(kernel=kernel):
                gamma, keywords = self.model(kernel=kernel, gamma="scale")._kernel_parameters(X)
                self.assertIsNone(gamma)
                self.assertEqual(keywords, {})

    def test_fit_and_prediction_use_dense_float64(self):
        X = (self.X * 10).astype(np.int32)
        model = self.fit_spy(self.model(sv_selection_ratio=1.0), X=X)
        for learner in RecordingDWD.instances:
            self.assertEqual(learner.X.dtype, np.float64)
        assert_array_equal(model.classes_, np.unique(self.y))
        self.assertEqual(model.n_features_in_, 2)
        model.decision_function(X[:3])
        self.assertEqual(model.final_estimator_.last_prediction_X.dtype, np.float64)

    def test_predict_delegates_to_final_margin_and_restores_labels(self):
        model = self.fit_spy()
        scores = model.final_estimator_.decision_function(self.X)
        assert_allclose(model.decision_function(self.X), scores)
        assert_array_equal(model.predict(self.X), model.classes_[(scores > 0).astype(int)])
        for labels in [np.tile([-7, 13], 10), np.tile([False, True], 10)]:
            other = self.fit_spy(self.model(sv_selection_ratio=1.0), y=labels)
            self.assertEqual(set(other.predict(self.X)), set(np.unique(labels)))

    def test_randomness_is_reproducible_without_global_rng_changes(self):
        np.random.seed(912)
        before = np.random.get_state()
        first = self.fit_spy()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        second = self.fit_spy()
        for a, b in zip(first.subset_indices_, second.subset_indices_):
            assert_array_equal(a, b)
        assert_array_equal(first.selected_indices_, second.selected_indices_)
        third = self.fit_spy(self.model(random_state=18))
        self.assertTrue(any(not np.array_equal(a, b)
                            for a, b in zip(first.subset_indices_, third.subset_indices_)))

    def test_clone_classifier_tags_and_parameter_assignment(self):
        model = self.model()
        copied = clone(model)
        self.assertEqual(copied.get_params(deep=True), model.get_params(deep=True))
        self.assertTrue(is_classifier(model))
        self.assertIs(copied.set_params(theta=0.2, lambd=0.4), copied)
        self.assertEqual(copied.theta, 0.2)
        self.assertEqual(copied.lambd, 0.4)
        self.assertFalse(hasattr(copied, "classes_"))

    def test_input_and_kernel_kwargs_are_not_mutated(self):
        X, y = self.X.copy(), self.y.copy()
        kws = {"degree": 2}
        model = self.fit_spy(self.model(kernel="poly", kernel_kws=kws,
                                       sv_selection_ratio=1.0), X=X, y=y)
        assert_array_equal(X, self.X)
        assert_array_equal(y, self.y)
        self.assertEqual(kws, {"degree": 2})
        self.assertEqual(model.kernel_kws, {"degree": 2})

    def test_invalid_parameters(self):
        invalid = {"M": [0, -1, 1.5, True], "theta": [0, -0.1, 1.1, np.nan],
                   "sv_selection_ratio": [0, -0.1, 1.1, np.nan],
                   "lambd": [0, -1, np.inf, np.nan], "q": [0, -1, np.nan],
                   "gamma": [-1, np.inf, np.nan, "wrong"],
                   "max_iter": [0, -1, 1.5, True],
                   "obj_tol": [-1, np.nan], "tol": [0, -1, np.inf],
                   "stopping": ["validation", "wrong"],
                   "acceleration": ["wrong", True],
                   "prediction_batch_size": [0, -1, True],
                   "kernel": ["precomputed"]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(parameter=key, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        self.fit_spy(self.model(**{key: value}))

    def test_positive_parameters_are_not_limited_to_search_grid(self):
        model = self.fit_spy(self.model(theta=0.35, sv_selection_ratio=0.85,
                                       gamma=3.7, lambd=0.03, q=2.5))
        self.assertEqual(model.gamma_, 3.7)
        self.assertEqual(len(model.selected_indices_), 17)

    def test_zero_gamma_is_explicitly_supported(self):
        model = self.fit_spy(self.model(gamma=0.0, sv_selection_ratio=1.0))
        self.assertEqual(model.gamma_, 0.0)
        self.assertTrue(all(learner.params["kernel_kws"]["gamma"] == 0.0
                            for learner in RecordingDWD.instances))

    def test_invalid_feature_and_label_inputs(self):
        bad_cases = [(self.X, np.zeros(20)),
                     (self.X, np.arange(20) % 3),
                     (self.X, self.y[:-1]),
                     (self.X.ravel(), self.y),
                     (np.where(self.X == self.X[0, 0], np.nan, self.X), self.y),
                     (sparse.csr_matrix(self.X), self.y)]
        for X, y in bad_cases:
            with self.subTest(shape=np.shape(X)), self.assertRaises((TypeError, ValueError)):
                self.fit_spy(X=X, y=y)

    def test_ineligible_selection_does_not_fit_a_fallback(self):
        for ratio, y in [(0.05, self.y),
                         (0.25, np.array(["left"] * 10 + ["right"] * 10))]:
            ConstantDWD.instances = []
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.fit_spy(self.model(sv_selection_ratio=ratio), y=y, spy=ConstantDWD)
            self.assertLessEqual(len(ConstantDWD.instances), 3)

    def test_failed_refit_clears_previous_fitted_state(self):
        model = self.fit_spy()
        RecordingDWD.fail_at = len(RecordingDWD.instances)
        with self.assertRaises(FloatingPointError):
            self.fit_spy(model)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)
        for name in ["final_estimator_", "estimators_", "classes_",
                     "selected_indices_", "consensus_scores_", "n_iter_", "converged_"]:
            self.assertFalse(hasattr(model, name), name)

    def test_validation_and_final_fit_failures_also_clear_state(self):
        model = self.fit_spy()
        with self.assertRaises(ValueError):
            self.fit_spy(model, y=np.zeros(len(self.y)))
        self.assertFalse(hasattr(model, "n_features_in_"))
        self.assertFalse(hasattr(model, "final_estimator_"))
        model = self.fit_spy()
        RecordingDWD.fail_at = len(RecordingDWD.instances) + model.M
        with self.assertRaises(FloatingPointError):
            self.fit_spy(model)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_nonfinite_or_wrongly_shaped_base_scores_fail_explicitly(self):
        for values in [np.full(20, np.nan), np.full(20, np.inf), np.zeros((20, 2))]:
            class BrokenScores(RecordingDWD):
                def decision_function(self, X):
                    return values
            with self.subTest(shape=values.shape), self.assertRaises(FloatingPointError):
                self.fit_spy(spy=BrokenScores)

    def test_supplied_random_state_is_copied(self):
        rng = np.random.RandomState(83)
        before = rng.get_state()
        first = self.fit_spy(self.model(random_state=rng))
        after = rng.get_state()
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        second = self.fit_spy(self.model(random_state=rng))
        for a, b in zip(first.subset_indices_, second.subset_indices_):
            assert_array_equal(a, b)

    def test_feature_names_when_dataframe_support_is_available(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("Optional pandas is not installed")
        X = pd.DataFrame(self.X, columns=["first", "second"])
        model = self.fit_spy(X=X)
        assert_array_equal(model.feature_names_in_, ["first", "second"])
        self.assertEqual(model.predict(X).shape, (20,))
        with self.assertRaises(ValueError):
            model.predict(X[["second", "first"]])

    def test_unfitted_and_wrong_feature_prediction_fail(self):
        model = self.model()
        with self.assertRaises(NotFittedError):
            model.predict(self.X)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        self.fit_spy(model)
        with self.assertRaises(ValueError):
            model.decision_function(np.ones((3, 3)))


class ConditionalSamplingTests(unittest.TestCase):
    def test_draws_have_exact_size_unique_rows_and_both_classes(self):
        y = np.array([0] * 49 + [1])
        subsets = _sample_subsets(y, 7, 20, np.random.RandomState(123))
        self.assertEqual(len(subsets), 20)
        for subset in subsets:
            self.assertEqual(len(subset), 7)
            self.assertEqual(len(np.unique(subset)), 7)
            self.assertEqual(set(y[subset]), {0, 1})
            self.assertTrue(np.all((subset >= 0) & (subset < len(y))))

    def test_small_population_matches_uniform_conditional_distribution(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        possible = {tuple(c) for c in itertools.combinations(range(6), 3)
                    if len(set(y[list(c)])) == 2}
        subsets = _sample_subsets(y, 3, 1600, np.random.RandomState(1234))
        counts = Counter(tuple(sorted(subset)) for subset in subsets)
        self.assertEqual(set(counts), possible)
        # Each of the 16 eligible subsets has probability 1/16. These broad
        # deterministic-sample bounds detect a forced equal class allocation.
        self.assertTrue(all(50 < count < 150 for count in counts.values()), counts)
        one_positive = sum(int(np.sum(y[s]) == 1) for s in subsets)
        self.assertTrue(1120 < one_positive < 1280, one_positive)

    def test_full_population_draws_and_reproducibility(self):
        y = np.array([0, 1, 0, 1])
        first = _sample_subsets(y, 4, 3, np.random.RandomState(55))
        second = _sample_subsets(y, 4, 3, np.random.RandomState(55))
        for a, b in zip(first, second):
            assert_array_equal(np.sort(a), np.arange(4))
            assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
