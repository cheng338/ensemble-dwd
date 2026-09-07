"""Small real-solver integration checks; no downloaded datasets are required."""

import pickle
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from ensemble_dwd import EnsembleKernelDWD


def affine_matrix_kernel(X, Y, scale=1.0, offset=0.0):
    """Matrix-level callable kernel, as required by the DWD dependency."""
    return scale * (X @ Y.T) + offset


def shared_gamma_score(estimator, X, y):
    """Expose each fold's resolved gamma while checking every fitted learner."""
    model = estimator["ensemble"]
    for learner in [*model.estimators_, model.final_estimator_]:
        if learner.kernel_kws["gamma"] != model.gamma_:
            raise AssertionError("A learner resolved a different kernel coefficient")
    return model.gamma_


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = make_classification(n_samples=48, n_features=4, n_informative=3,
                                      n_redundant=0, class_sep=1.5, random_state=91)
        cls.y = np.where(y == 0, "negative", "positive")

    def estimator(self):
        return EnsembleKernelDWD(M=2, theta=0.5, sv_selection_ratio=1.0,
                                 lambd=0.05, gamma=0.2, random_state=7,
                                 max_iter=12, stopping="fixed")

    def test_sklearn_estimator_contract(self):
        # Full retention makes all generated binary fixtures eligible without
        # altering the estimator's ordinary sampling or final-fit procedure.
        # The framework handles optional checks according to installed extras.
        check_estimator(EnsembleKernelDWD(M=2, theta=0.5, sv_selection_ratio=1.0,
                                         lambd=0.05, gamma=0.2), on_skip=None)

    def test_real_solver_finite_predictions_and_shared_parameters(self):
        model = self.estimator().fit(self.X, self.y)
        self.assertTrue(is_classifier(model))
        self.assertEqual(model.decision_function(self.X).shape, (48,))
        self.assertTrue(np.isfinite(model.decision_function(self.X)).all())
        self.assertTrue(set(model.predict(self.X)).issubset(set(self.y)))
        self.assertGreater(model.score(self.X, self.y), 0.65)
        for learner in [*model.estimators_, model.final_estimator_]:
            self.assertEqual(learner.implementation, "optimized")
            self.assertEqual(learner.lambd, model.lambd)
            self.assertEqual(learner.q, model.q)
            self.assertEqual(learner.kernel_kws["gamma"], model.gamma_)
        self.assertEqual(len(model.n_iter_), 3)
        self.assertTrue(np.all(np.asarray(model.n_iter_) <= 12))

    def test_pipeline_gridsearch_and_manual_fold_scores_agree(self):
        pipeline = Pipeline([("scale", StandardScaler()), ("ensemble", self.estimator())])
        cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=11).split(self.X, self.y))
        search = GridSearchCV(pipeline, {"ensemble__lambd": [0.02, 0.1]},
                              cv=cv, scoring="accuracy", n_jobs=1, error_score="raise")
        search.fit(self.X, self.y)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        expected = []
        for params in search.cv_results_["params"]:
            fold_scores = []
            for train, validation in cv:
                candidate = clone(pipeline).set_params(**params)
                candidate.fit(self.X[train], self.y[train])
                fold_scores.append(candidate.score(self.X[validation], self.y[validation]))
            expected.append(np.mean(fold_scores))
        assert_allclose(search.cv_results_["mean_test_score"], expected, rtol=0, atol=0)
        self.assertTrue(is_classifier(search.best_estimator_))
        assert_allclose(search.best_estimator_["scale"].mean_, self.X.mean(axis=0))
        self.assertEqual(search.predict(self.X[:3]).shape, (3,))

    def test_pickle_and_joblib_roundtrip(self):
        model = self.estimator().fit(self.X, self.y)
        expected_scores = model.decision_function(self.X)
        expected_labels = model.predict(self.X)
        restored = pickle.loads(pickle.dumps(model))
        assert_array_equal(restored.predict(self.X), expected_labels)
        assert_allclose(restored.decision_function(self.X), expected_scores, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ensemble.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            assert_array_equal(restored.predict(self.X), expected_labels)
            assert_allclose(restored.decision_function(self.X), expected_scores, rtol=0, atol=0)

    def test_multiworker_gridsearch_matches_serial_and_uses_fold_gamma(self):
        # Centering without variance scaling leaves a different feature variance
        # in each fold, so a coefficient computed from the entire dataset would
        # be observable. The ordinary standardized pipeline is tested above.
        pipeline = Pipeline([
            ("scale", StandardScaler(with_std=False)),
            ("ensemble", self.estimator().set_params(gamma="scale", max_iter=6)),
        ])
        cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=29)
                  .split(self.X, self.y))
        serial = GridSearchCV(
            pipeline, {"ensemble__lambd": [0.03, 0.1]}, cv=cv,
            scoring={"accuracy": "accuracy", "resolved_gamma": shared_gamma_score},
            refit="accuracy", error_score="raise", n_jobs=1, pre_dispatch=2,
        ).fit(self.X, self.y)
        parallel = clone(serial).set_params(n_jobs=2)
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            parallel.fit(self.X, self.y)
        for key in ["mean_test_accuracy", "mean_test_resolved_gamma",
                    "rank_test_accuracy"]:
            assert_allclose(parallel.cv_results_[key], serial.cv_results_[key],
                            rtol=0, atol=0)
        expected_gammas = []
        for fold, (train, _) in enumerate(cv):
            centered = self.X[train] - self.X[train].mean(axis=0)
            expected = 1.0 / (self.X.shape[1] * centered.var())
            expected_gammas.append(expected)
            key = f"split{fold}_test_resolved_gamma"
            assert_allclose(serial.cv_results_[key], expected, rtol=1e-14, atol=0)
            assert_allclose(parallel.cv_results_[key], expected, rtol=1e-14, atol=0)
        self.assertGreater(np.ptp(expected_gammas), 1e-5)
        assert_array_equal(parallel.predict(self.X), serial.predict(self.X))
        self.assertEqual(parallel.best_params_, serial.best_params_)

    def test_real_stopping_policies_and_generalized_loss_are_forwarded(self):
        for stopping in ["fixed", "objective", "optimality"]:
            for q in [0.5, 2.0]:
                with self.subTest(stopping=stopping, q=q):
                    model = self.estimator().set_params(
                        q=q, stopping=stopping, max_iter=8, obj_tol=1e-4, tol=1e-5,
                    ).fit(self.X, self.y)
                    self.assertTrue(np.isfinite(model.decision_function(self.X)).all())
                    for learner in [*model.estimators_, model.final_estimator_]:
                        self.assertEqual(learner.q, q)
                        self.assertEqual(learner.stopping, stopping)
                        self.assertEqual(learner.obj_tol, 1e-4)
                        self.assertEqual(learner.tol, 1e-5)
                        self.assertLessEqual(learner.n_iter_, 8)
                        if stopping == "fixed":
                            self.assertEqual(learner.n_iter_, 8)

    def test_linear_callable_and_polynomial_kernels(self):
        X = StandardScaler().fit_transform(self.X)
        linear = self.estimator().set_params(kernel="linear").fit(X, self.y)
        callable_linear = self.estimator().set_params(
            kernel=affine_matrix_kernel, kernel_kws={"scale": 1.0, "offset": 0.0},
        ).fit(X, self.y)
        assert_allclose(callable_linear.decision_function(X), linear.decision_function(X),
                        rtol=1e-8, atol=1e-9)
        assert_array_equal(callable_linear.predict(X), linear.predict(X))
        self.assertEqual(linear.kernel_kws_, {})
        self.assertEqual(callable_linear.kernel_kws_, {"scale": 1.0, "offset": 0.0})
        for model in [linear, callable_linear]:
            self.assertTrue(np.isfinite(model.decision_function(X)).all())
            self.assertGreater(model.score(X, self.y), 0.65)
        keywords = {"degree": 2, "coef0": 0.5}
        polynomial = self.estimator().set_params(kernel="poly", kernel_kws=keywords).fit(X, self.y)
        self.assertTrue(np.isfinite(polynomial.decision_function(X)).all())
        for learner in [*polynomial.estimators_, polynomial.final_estimator_]:
            self.assertEqual(learner.kernel_kws, {"degree": 2, "coef0": 0.5, "gamma": 0.2})
        self.assertEqual(keywords, {"degree": 2, "coef0": 0.5})


if __name__ == "__main__":
    unittest.main()
