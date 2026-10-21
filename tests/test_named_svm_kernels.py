"""Named SVM kernels preserve selection, estimator and checkpoint contracts."""

from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ensemble_dwd import EnsembleKernelSVM, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint


def fixture(seed=13, n=96):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.where(X[:, 0] + .15 * X[:, 1] ** 2 > 0, 'positive', 'negative')
    return X, y, rng.normal(size=(19, 6))


class NamedKernelSVMTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.query = fixture()

    def model(self, **changes):
        options = dict(M=3, theta=.2, sv_selection_ratio=.6,
                       random_state=17, tol=1e-5)
        options.update(changes)
        return EnsembleKernelSVM(**options)

    def test_manual_svc_reconstruction_for_all_named_kernels(self):
        for kernel in ('rbf', 'linear', 'poly'):
            for seed in (13, 27, 49):
                with self.subTest(kernel=kernel, seed=seed):
                    X, y, query = fixture(seed)
                    model = self.model(kernel=kernel, degree=2, coef0=.5).fit(X, y)
                    options = dict(C=model.C, gamma=model.gamma_, kernel=kernel,
                                   degree=2, coef0=.5, tol=model.tol, random_state=0)
                    bases = [SVC(**options).fit(X[indices], y[indices])
                             for indices in model.subset_indices_]
                    mean = np.sum([base.decision_function(X) for base in bases], axis=0) / model.M
                    signed = np.where(y == model.classes_[1], 1., -1.)
                    selected = np.array(sorted(
                        range(len(y)), key=lambda i: (abs(signed[i] * mean[i] - 1), i)
                    ))[:int(.6 * len(y))]
                    final = SVC(**options).fit(X[selected], y[selected])
                    assert_array_equal(mean, model.consensus_scores_)
                    assert_array_equal(selected, model.selected_indices_)
                    assert_array_equal(final.decision_function(query), model.decision_function(query))
                    self.assertEqual(clone(model).get_params(), model.get_params())
                    assert_array_equal(pickle.loads(pickle.dumps(model)).predict(query), model.predict(query))
                    self.assertEqual(sum(map(len, model.subset_indices_)),
                                     len(np.unique(np.concatenate(model.subset_indices_))))

    def test_kernel_options_have_separate_cache_identities(self):
        for kernel in ('rbf', 'linear', 'poly'):
            with self.subTest(kernel=kernel), tempfile.TemporaryDirectory() as cache:
                first = self.model(kernel=kernel, degree=2, coef0=.5,
                                   cache_dir=cache).fit(self.X, self.y)
                second = clone(first).set_params(sv_selection_ratio=.7).fit(self.X, self.y)
                self.assertFalse(first.cache_hit_)
                self.assertTrue(second.cache_hit_, second.cache_status_)
                self.assertEqual(first.cache_key_, second.cache_key_)
                assert_array_equal(first.consensus_scores_, second.consensus_scores_)
                self.assertIsNot(first.final_estimator_, second.final_estimator_)
                for changes in ({'degree': 3}, {'coef0': .75},
                                {'kernel': 'linear' if kernel != 'linear' else 'rbf'}):
                    changed = clone(first).set_params(**changes).fit(self.X, self.y)
                    self.assertFalse(changed.cache_hit_)
                    self.assertNotEqual(first.cache_key_, changed.cache_key_)
                lean = clone(first).set_params(store_base_estimators=False).fit(self.X, self.y)
                self.assertIsNone(lean.estimators_)

    def test_default_and_explicit_batching_cross_training_and_query_boundary(self):
        rng = np.random.RandomState(20261015)
        X = rng.normal(size=(1025, 8))
        query = rng.normal(size=(1025, 8))
        y = np.where(X[:, 0] + .2 * X[:, 1] > 0, 1, -1)
        original_decision = SVC.decision_function
        with threadpool_limits(limits=1):
            for kernel in ('rbf', 'linear', 'poly'):
                models = []
                for batch in (None, 1024):
                    with self.subTest(kernel=kernel, batch=batch):
                        model = EnsembleKernelSVM(
                            M=5, theta=.1, sv_selection_ratio=.3, kernel=kernel,
                            C=1., gamma=.125, random_state=20261015)
                        if batch is not None:
                            model.set_params(prediction_batch_size=batch)
                        calls = []

                        def observed(child, values):
                            calls.append(len(values))
                            return original_decision(child, values)

                        with patch.object(SVC, 'decision_function', observed):
                            model.fit(X, y)
                        self.assertEqual(calls, [1025] * 5 if batch is None else [1024, 1] * 5)
                        manual = np.zeros(len(X))
                        for child in model.estimators_:
                            manual += child.decision_function(X)
                        manual /= model.M
                        selected = np.argsort(abs(y * manual - 1.), kind='stable')[:307]
                        assert_array_equal(manual, model.consensus_scores_)
                        assert_array_equal(selected, model.selected_indices_)
                        for n in (1023, 1024, 1025):
                            calls.clear()
                            with patch.object(SVC, 'decision_function', observed):
                                scores = model.decision_function(query[:n])
                            self.assertEqual(calls, [n] if batch is None or n <= 1024 else [1024, 1])
                            assert_array_equal(scores, model.final_estimator_.decision_function(query[:n]))
                            assert_array_equal(model.predict(query[:n]), model.final_estimator_.predict(query[:n]))
                        models.append(model)
                full, batched = models
                assert_array_equal(full.consensus_scores_, batched.consensus_scores_)
                assert_array_equal(full.selected_indices_, batched.selected_indices_)
                for left, right in zip(full.subset_indices_, batched.subset_indices_):
                    assert_array_equal(left, right)
                for left, right in zip([*full.estimators_, full.final_estimator_],
                                       [*batched.estimators_, batched.final_estimator_]):
                    for name in ('classes_', 'support_', 'support_vectors_', 'dual_coef_',
                                 'intercept_', '_dual_coef_', '_intercept_', 'n_iter_'):
                        assert_array_equal(getattr(left, name), getattr(right, name))

    def test_invalid_named_kernel_options_clear_fitted_state(self):
        for changes in ({'kernel': 'sigmoid'}, {'kernel': lambda a, b: a @ b.T},
                        {'degree': 0}, {'degree': True}, {'degree': 2.5},
                        {'coef0': -1}, {'coef0': complex(1, 0)}, {'coef0': np.inf}):
            with self.subTest(changes=str(changes)):
                model = self.model().fit(self.X, self.y)
                model.set_params(**changes)
                with self.assertRaises((ValueError, TypeError)):
                    model.fit(self.X, self.y)
                with self.assertRaises(NotFittedError):
                    model.predict(self.query)

    def test_multiclass_and_training_only_pipeline_cv(self):
        X, y, _ = fixture(n=90)
        labels = np.arange(90) % 3
        ovo = OvOClassifier(self.model(M=2, theta=.25, sv_selection_ratio=.8,
                                      kernel='poly', coef0=1)).fit(X, labels)
        self.assertEqual(ovo.predict(self.query).shape, (19,))
        self.assertTrue(all(child.kernel == 'poly' for child in ovo.estimators_))
        model = make_pipeline(StandardScaler(), self.model(
            M=2, theta=.25, sv_selection_ratio=.8))
        search = GridSearchCV(
            model, {'ensemblekernelsvm__kernel': ['linear', 'poly'],
                    'ensemblekernelsvm__degree': [2],
                    'ensemblekernelsvm__coef0': [.5, 1.]},
            cv=StratifiedKFold(3, shuffle=True, random_state=8),
            error_score='raise', n_jobs=1).fit(X, y)
        self.assertEqual(len(search.cv_results_['params']), 4)

    def test_base_failure_and_interruption_clear_named_kernel_fit(self):
        original = SVC.fit
        for error in (RuntimeError('controlled failure'), KeyboardInterrupt('controlled interrupt')):
            with self.subTest(error=type(error).__name__):
                model = self.model(kernel='poly').fit(self.X, self.y)
                calls = []

                def failing(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 2:
                        raise error
                    return original(*args, **kwargs)

                with patch.object(SVC, 'fit', failing):
                    with self.assertRaises(type(error)):
                        model.fit(self.X, self.y)
                with self.assertRaises(NotFittedError):
                    model.predict(self.query)
                model.fit(self.X, self.y)
                self.assertEqual(model.predict(self.query).shape, (19,))

    def test_older_rbf_state_supplies_missing_kernel_options(self):
        model = self.model().fit(self.X, self.y)
        expected = model.decision_function(self.query)
        # Reproduce the absent fields of previously serialized RBF objects.
        del model.degree
        del model.coef0
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.degree, 3)
        self.assertEqual(restored.coef0, 0.)
        assert_array_equal(restored.decision_function(self.query), expected)
        copied = clone(restored).fit(self.X, self.y)
        assert_array_equal(copied.decision_function(self.query), expected)

    def test_completed_checkpoints_reuse_and_reject_changed_kernel_options(self):
        rng = np.random.default_rng(176)
        X = rng.normal(size=(72, 4))
        y = (X[:, 0] > .1).astype(int)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=27).split(X, y))
        for kernel in ('linear', 'poly', 'rbf'):
            with self.subTest(kernel=kernel), tempfile.TemporaryDirectory() as tmp:
                model = EnsembleKernelSVM(M=2, theta=.25, sv_selection_ratio=.8,
                                          kernel=kernel, degree=2, coef0=.5, random_state=4)
                options = dict(run_dir=Path(tmp) / 'checkpoint', estimator=model,
                               X=X, y=y, folds=folds, jobs=1, error_policy='raise',
                               provenance={'purpose': 'named-kernel-checkpoint-contract'})
                with checkpoint.CheckpointedCV(**options) as store:
                    first = store.search({'C': [.1, 1.]})
                self.assertEqual(first.checkpoint_summary_['new_folds'], 6)
                with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('Completed fold reran')):
                    with checkpoint.CheckpointedCV(**options, resume=True) as store:
                        again = store.search({'C': [1., .1]})
                self.assertEqual(again.checkpoint_summary_['reused_folds'], 6)
                for changes in ({'degree': 3}, {'coef0': 1.}, {'cache_size': 200.},
                                {'prediction_batch_size': 1024},
                                {'kernel': 'rbf' if kernel != 'rbf' else 'linear'}):
                    changed = clone(model).set_params(**changes)
                    with self.assertRaises(checkpoint.CheckpointError):
                        with checkpoint.CheckpointedCV(
                                **dict(options, estimator=changed), resume=True):
                            pass


if __name__ == '__main__':
    unittest.main()
