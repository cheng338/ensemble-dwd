"""Sampling-seed aggregation, immutable identity and partial-resume controls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
from ensemble_dwd import IneligibleSelectionError


class SeededClassifier(ClassifierMixin, BaseEstimator):
    events = []
    interrupt_at = None

    def __init__(self, offset=0., invalid_seed=None, random_state=42, label='a'):
        self.offset = offset
        self.invalid_seed = invalid_seed
        self.random_state = random_state
        self.label = label

    def fit(self, X, y):
        type(self).events.append((self.offset, self.random_state, self.label))
        if type(self).interrupt_at == len(type(self).events):
            raise KeyboardInterrupt('deliberate seed-fold interruption')
        if self.random_state == self.invalid_seed:
            raise IneligibleSelectionError('deliberate ineligible seed')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.threshold_ = self.offset + np.random.RandomState(self.random_state).uniform(-1., 1.)
        self.n_iter_ = self.random_state + 1
        return self

    def predict(self, X):
        return self.classes_[(np.asarray(X)[:, 0] > self.threshold_).astype(int)]


class ReplacingClassifier(ClassifierMixin, BaseEstimator):
    """A wrapper whose nonstandard setter can silently replace its child."""
    def __init__(self, estimator=None, swap=False):
        self.estimator = estimator
        self.swap = swap

    def set_params(self, **params):
        super().set_params(**params)
        if self.swap:
            self.estimator = SeededClassifier(random_state=self.estimator.random_state)
        return self


class MultiSeedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / 'run'
        self.X = np.array([[-4., 0.], [-3., 1.], [-2., 3.], [-1., 5.],
                           [1., 8.], [2., 13.], [3., 21.], [4., 34.]])
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.pipeline = Pipeline([('scale', StandardScaler()), ('model', SeededClassifier())])
        self.seeds = (0, 4, 8)
        self.grid = {'model__offset': [-.5, .5]}
        self.runtime = {'python': 'fixed-test-runtime', 'files_sha256': 'a' * 64,
                        'native_pools': [{'num_threads': 1, 'version': 'test-build'}]}
        runtime_patch = patch.object(checkpoint, 'runtime_fingerprint', side_effect=lambda **kwargs: deepcopy(self.runtime))
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        SeededClassifier.events = []
        SeededClassifier.interrupt_at = None

    def session(self, **kwargs):
        options = dict(run_dir=self.run, estimator=self.pipeline, X=self.X, y=self.y,
                       folds=self.folds, cv_seeds=self.seeds, seed_parameter='model__random_state')
        options.update(kwargs)
        store = checkpoint.CheckpointedCV(**options)
        # Production workflows create this directory through new_run_directory.
        # Create it after validation so invalid-constructor tests still prove
        # that no run files were written.
        (Path(options['run_dir']) / 'failures').mkdir(parents=True, exist_ok=True)
        return store

    def test_validation_normalizes_without_mutating_input(self):
        for values in ([0, np.int64(4)], (0, 4), np.array([0, 4], dtype=np.uint64)):
            actual = checkpoint.validate_cv_seeds(values)
            self.assertEqual(actual, (0, 4))
            self.assertTrue(all(type(seed) is int for seed in actual))
        values = [0, 4]
        with self.session(cv_seeds=values) as store:
            values.append(8)
            self.assertEqual(store.cv_seeds, (0, 4))
            self.assertEqual(store.identity['seed_policy']['seeds'], [0, 4])
        self.assertIsNone(checkpoint.validate_cv_seeds(None))
        self.assertEqual(checkpoint.validate_seed(np.uint64(2 ** 32 - 1)), 2 ** 32 - 1)

    def test_invalid_seeds_are_rejected_before_writing(self):
        values = [[], (), [0, 0], [True], [np.bool_(False)], [-1], [2 ** 32], [1.],
                  {0, 1}, '1,2', iter([1, 2]), np.array([[1, 2]]),
                  np.array([1.]), np.array([1], dtype=object), np.array([True])]
        for seeds in values:
            with self.subTest(seeds=repr(seeds)), self.assertRaises(ValueError):
                self.session(cv_seeds=seeds)
        self.assertFalse(self.run.exists())

    def test_seed_parameter_must_be_an_existing_integer_random_state(self):
        for name in (None, '', 'model__offset', 'missing__random_state', 'model____random_state'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.session(seed_parameter=name)
        with self.assertRaises(ValueError):
            self.session(cv_seeds=None)
        for seed in (None, True, 1.5, np.random.RandomState(1)):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.session(estimator=clone(self.pipeline).set_params(model__random_state=seed))
        self.assertFalse(self.run.exists())

    def test_explicit_manual_loop_and_timing_aggregation(self):
        candidates = list(ParameterGrid(self.grid))
        expected = np.empty((len(candidates), len(self.seeds), len(self.folds)))
        for ci, parameters in enumerate(candidates):
            for si, seed in enumerate(self.seeds):
                for fi, (train, validation) in enumerate(self.folds):
                    model = clone(self.pipeline).set_params(**parameters, model__random_state=seed)
                    model.fit(self.X[train], self.y[train])
                    expected[ci, si, fi] = accuracy_score(self.y[validation], model.predict(self.X[validation]))
        SeededClassifier.events = []
        with self.session() as store:
            result = store.search(self.grid)
        np.testing.assert_array_equal(result.seed_results_['scores'], expected)
        fold_means = expected.mean(axis=1)
        for fi in range(len(self.folds)):
            np.testing.assert_array_equal(result.cv_results_[f'split{fi}_test_score'], fold_means[:, fi])
        np.testing.assert_array_equal(result.cv_results_['mean_test_score'], fold_means.mean(axis=1))
        np.testing.assert_array_equal(result.cv_results_['std_test_score'], fold_means.std(axis=1))
        for singular, plural in (('fit_time', 'fit_times'), ('score_time', 'score_times')):
            work = result.seed_results_[plural].sum(axis=1)
            np.testing.assert_array_equal(result.cv_results_['mean_' + singular], work.mean(axis=1))
            np.testing.assert_array_equal(result.cv_results_['std_' + singular], work.std(axis=1))
        self.assertEqual(result.n_splits_, 2)
        self.assertEqual(result.checkpoint_summary_['candidate_count'], 2)
        self.assertEqual(result.checkpoint_summary_['evaluation_count'], 12)
        self.assertEqual(result.checkpoint_summary_['new_folds'], 12)
        self.assertEqual(result.checkpoint_summary_['executed_folds'], 12)
        self.assertEqual(result.seed_results_['axes'], ['candidate', 'seed', 'fold'])
        self.assertNotIn('estimator__model__random_state', result.cv_results_['params'][0])
        self.assertEqual(result.seed_results_['candidate_params'], candidates)
        self.assertEqual([row['new_folds'] for row in result.checkpoint_summary_['seed_counts']], [4, 4, 4])
        self.assertEqual(self.pipeline.get_params()['model__random_state'], 42)

    def test_actual_receipts_and_diagnostics_identify_every_seed(self):
        with self.session() as store:
            result = store.search({'model__offset': [0.]})
        task_ids = result.seed_results_['receipt_task_sha256'][0]
        self.assertEqual(len({task for row in task_ids for task in row}), 6)
        for si, seed in enumerate(self.seeds):
            for fi, task in enumerate(task_ids[si]):
                record = json.loads((self.run / 'cv-checkpoints/folds' / (task + '.json')).read_text())['payload']
                self.assertEqual(record['parameters'], checkpoint._stable({'model__offset': 0., 'model__random_state': seed}))
                self.assertEqual(result.seed_results_['receipt_parameters'][0][si][fi], record['parameters'])
                self.assertEqual(result.seed_results_['diagnostics'][0][si][fi]['n_iter_'], seed + 1)
                self.assertEqual(record['fold'], fi)

    def test_ties_keep_first_candidate_and_seed_is_not_a_candidate_label(self):
        with self.session() as store:
            result = store.search({'model__label': ['first', 'second']})
        self.assertEqual(result.best_index_, 0)
        self.assertEqual(result.best_params_, {'estimator__model__label': 'first'})
        np.testing.assert_array_equal(result.cv_results_['rank_test_score'], [1, 1])
        self.assertEqual(result.cv_results_['param_estimator__model__label'].tolist(), ['first', 'second'])

    def test_ineligible_seed_invalidates_candidate_without_nanmean(self):
        with self.session() as store:
            result = store.search({'model__invalid_seed': [4, None]})
        self.assertTrue(np.isnan(result.cv_results_['mean_test_score'][0]))
        self.assertTrue(np.isnan(result.cv_results_['split0_test_score'][0]))
        self.assertTrue(np.isfinite(result.cv_results_['mean_test_score'][1]))
        self.assertTrue(np.isnan(result.seed_results_['scores'][0, 1]).all())
        self.assertTrue(np.isfinite(result.seed_results_['scores'][0, [0, 2]]).all())
        self.assertEqual(result.best_index_, 1)
        self.assertEqual(result.checkpoint_summary_['new_folds'], 12)
        failures = {path.name: path.read_bytes() for path in (self.run / 'failures').glob('*.json')}
        self.assertEqual(len(failures), len(self.folds))
        for content in failures.values():
            record = json.loads(content)
            self.assertEqual(record['category'], 'ineligible_selection')
            self.assertEqual(record['phase'], 'cv_fit')
            self.assertEqual(record['parameters']['model__random_state'], 4)
        SeededClassifier.events = []
        with self.session(resume=True) as store:
            resumed = store.search({'model__invalid_seed': [4, None]})
        self.assertEqual(SeededClassifier.events, [])
        self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 12)
        np.testing.assert_array_equal(resumed.cv_results_['mean_test_score'], result.cv_results_['mean_test_score'])
        self.assertEqual({path.name: path.read_bytes() for path in (self.run / 'failures').glob('*.json')}, failures)

    def test_seed_or_ancestor_replacement_is_rejected_before_any_fit(self):
        alternatives = [
            {'model__random_state': [1]}, {'model': [SeededClassifier()]},
            {'steps': [[('scale', StandardScaler()), ('model', SeededClassifier())]]},
        ]
        with self.session() as store:
            for grid in alternatives:
                with self.subTest(keys=list(grid)), self.assertRaises(ValueError):
                    store.search(grid)
        self.assertEqual(SeededClassifier.events, [])
        self.assertFalse(list((self.run / 'cv-checkpoints/folds').glob('*.json')))

    def test_nonstandard_setter_cannot_replace_seeded_estimator(self):
        wrapper = ReplacingClassifier(estimator=SeededClassifier())
        with self.session(estimator=wrapper, seed_parameter='estimator__random_state') as store:
            with self.assertRaisesRegex(ValueError, 'sampling-seed path'):
                store.search({'swap': [True]})
        self.assertEqual(SeededClassifier.events, [])

    def test_interrupt_and_resume_only_unfinished_seed_folds(self):
        SeededClassifier.interrupt_at = 4
        with self.assertRaises(KeyboardInterrupt), self.session() as store:
            store.search(self.grid)
        saved = list((self.run / 'cv-checkpoints/folds').glob('*.json'))
        self.assertEqual(len(saved), 3)
        before = {path.name: path.read_bytes() for path in saved}
        SeededClassifier.events = []
        SeededClassifier.interrupt_at = None
        with self.session(resume=True) as store:
            result = store.search(self.grid)
        self.assertEqual(len(SeededClassifier.events), 9)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 3)
        self.assertEqual(result.checkpoint_summary_['new_folds'], 9)
        self.assertEqual([row['reused_folds'] for row in result.checkpoint_summary_['seed_counts']], [2, 1, 0])
        for path in saved:
            self.assertEqual(path.read_bytes(), before[path.name])
        attempts = [json.loads(path.read_text())['payload']['state']
                    for path in (self.run / 'cv-checkpoints/attempts').glob('*.json')]
        self.assertIn('interrupted', attempts)

    def test_complete_seed_policy_must_match_on_resume(self):
        with self.session():
            pass
        for seeds in ([8, 4, 0], [0, 4], [0, 4, 9], [0]):
            with self.subTest(seeds=seeds), self.assertRaises(checkpoint.CheckpointError):
                with self.session(resume=True, cv_seeds=seeds):
                    pass
        with self.assertRaises(checkpoint.CheckpointError):
            with self.session(resume=True, cv_seeds=None, seed_parameter=None):
                pass

    def test_changed_seed_policy_in_active_context_is_rejected(self):
        with self.session() as store:
            store.cv_seeds = (8, 4, 0)
            with self.assertRaises(checkpoint.CheckpointError):
                store.search(self.grid)
            store.cv_seeds = None
            with self.assertRaises(checkpoint.CheckpointError):
                store.search(self.grid)
            store.cv_seeds = self.seeds
            store.seed_parameter = 'random_state'
            with self.assertRaises(checkpoint.CheckpointError):
                store.search(self.grid)
        self.assertEqual(SeededClassifier.events, [])

    def test_one_explicit_seed_matches_old_path_and_old_results_stay_unextended(self):
        pipeline = clone(self.pipeline).set_params(model__random_state=4)
        with self.session(estimator=pipeline, cv_seeds=None, seed_parameter=None,
                          reuse_equivalent_folds=False) as store:
            original = store.search(self.grid)
            self.assertNotIn('seed_policy', store.identity)
        with self.session(run_dir=self.run.parent / 'explicit', estimator=pipeline, cv_seeds=[4],
                          reuse_equivalent_folds=False) as store:
            explicit = store.search(self.grid)
        for key in ('params', 'mean_test_score', 'std_test_score', 'split0_test_score',
                    'split1_test_score', 'rank_test_score', 'param_estimator__model__offset'):
            np.testing.assert_equal(explicit.cv_results_[key], original.cv_results_[key])
        self.assertEqual(original.best_params_, explicit.best_params_)
        self.assertEqual(original.n_splits_, explicit.n_splits_)
        self.assertFalse(hasattr(original, 'seed_results_'))
        self.assertNotIn('seed_policy', original.checkpoint_summary_)
        self.assertNotIn('executed_folds', original.checkpoint_summary_)


if __name__ == '__main__':
    unittest.main()
