"""Exact retained-count reuse, numerical parity, and durable source controls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint


class CustomDWD(EnsembleKernelDWD):
    """An otherwise unchanged subclass must not inherit the equivalence proof."""


class CustomScaler(StandardScaler):
    """A custom transformer is outside the built-in preprocessing contract."""


class CustomFloat(float):
    """Scalar subclasses are outside the built-in arithmetic contract."""


class CustomDict(dict):
    """Container subclasses need not preserve ordinary parameter semantics."""


def binary_data():
    # Interleaved, balanced classes with distinct distances from the boundary.
    x = np.linspace(.3, 2.0, 16)
    X = np.column_stack((np.ravel(np.column_stack((-x, x))),
                         np.repeat(np.sin(x), 2)))
    return X, np.tile([0, 1], 16)


def multiclass_data():
    centers = np.array([[-2., -1.], [2., -1.], [0., 2.]])
    offsets = np.array([[-.12, 0.], [.12, 0.], [0., -.12], [0., .12],
                        [-.08, -.08], [.08, .08], [-.08, .08], [.08, -.08]])
    return np.vstack([center + offsets for center in centers]), np.repeat([0, 1, 2], 8)


def pipeline(kind='dwd', *, multiclass=False):
    options = dict(M=1, theta=1., gamma=.2, random_state=42,
                   sv_selection_ratio=.875, store_base_estimators=False)
    model = (EnsembleKernelDWD(lambd=.1, max_iter=4, stopping='fixed', **options)
             if kind == 'dwd' else EnsembleKernelSVM(C=1., **options))
    if multiclass:
        model = OvOClassifier(model, n_jobs=1)
    return Pipeline([('scale', StandardScaler()), ('model', model)])


class FoldEquivalenceProofTests(unittest.TestCase):
    """Boundary and scope checks need no estimator fitting or runtime scan."""

    def signature(self, model, labels, *, identity='a' * 64, fold=0):
        proof = checkpoint._fold_equivalence(model, identity, fold, np.asarray(labels))
        return None if proof is None else proof['key']

    def test_malformed_typed_ratio_and_duplicate_parameter_keys_are_rejected(self):
        model = pipeline().named_steps['model']
        for encoded_ratio in (['int', .875], ['float', .875], ['float', '0.875'], ['int']):
            identity = checkpoint._stable(model)
            parameters = dict(identity[2][1])
            parameters['sv_selection_ratio'][:] = encoded_ratio
            with self.subTest(ratio=encoded_ratio), self.assertRaises(checkpoint.CheckpointError):
                checkpoint._retention_identity(identity)
        identity = checkpoint._stable(model)
        identity[2][1].append(identity[2][1][0])
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint._retention_identity(identity)

    def test_audited_close_ratios_match_both_training_sizes(self):
        first, second = .34101562499999993, .3412109375
        self.assertNotEqual(first, second)
        for n in (1933, 1934):
            with self.subTest(n=n):
                a = pipeline().set_params(model__sv_selection_ratio=first)
                b = pipeline().set_params(model__sv_selection_ratio=second)
                labels = np.arange(n) % 2
                self.assertEqual(int(n * first), 659)
                self.assertEqual(int(n * second), 659)
                signature = self.signature(a, labels)
                self.assertIsNotNone(signature)
                self.assertEqual(signature, self.signature(b, labels))
                self.assertNotEqual(signature, self.signature(b, labels, fold=1))
                self.assertNotEqual(signature, self.signature(b, labels, identity='b' * 64))

    def test_binary_count_boundary_and_all_other_controls_remain_distinct(self):
        labels = np.arange(16) % 2
        base = pipeline()
        expected = self.signature(base, labels)
        self.assertIsNotNone(expected)
        changes = {'model__sv_selection_ratio': .94, 'model__lambd': .2,
                   'model__gamma': .3, 'model__random_state': 43,
                   'model__max_iter': 5, 'model__obj_tol': 1e-7,
                   'model__residual_check_order': 'adaptive',
                   'scale__with_std': False, 'scale__with_mean': False}
        for name, value in changes.items():
            with self.subTest(parameter=name):
                changed = clone(base).set_params(**{name: value})
                self.assertNotEqual(expected, self.signature(changed, labels))
        svm = pipeline('svm')
        for name, value in {'model__C': 2., 'model__tol': .01,
                            'model__shrinking': False}.items():
            with self.subTest(parameter=name):
                self.assertNotEqual(self.signature(svm, labels),
                                    self.signature(clone(svm).set_params(**{name: value}), labels))

    def test_numpy_theta_rounding_cannot_alias_python_float_subset_count(self):
        theta = np.float32(.7)
        plain_theta = float(theta)
        # NumPy scalar promotion differs between supported dependency versions.
        # The equivalence proof must reject this type even when counts coincide.
        self.assertNotEqual(checkpoint._stable(theta), checkpoint._stable(plain_theta))
        labels = np.arange(10) % 2
        numpy_model = pipeline().set_params(model__theta=theta, model__sv_selection_ratio=.8)
        plain_model = pipeline().set_params(model__theta=plain_theta, model__sv_selection_ratio=.85)
        self.assertEqual(int(10 * .8), int(10 * .85))
        self.assertIsNone(self.signature(numpy_model, labels))
        plain_key = self.signature(plain_model, labels)
        self.assertIsNotNone(plain_key)
        self.assertEqual(plain_key, self.signature(
            clone(plain_model).set_params(model__sv_selection_ratio=.8), labels))

    def test_nested_numpy_and_custom_parameter_types_decline_equivalence(self):
        labels = np.arange(16) % 2
        changes = [dict(model__M=np.int64(1)),
                   dict(model__gamma=np.float64(.2)),
                   dict(scale__with_std=np.bool_(True)),
                   dict(model__kernel='poly', model__kernel_kws={'coef0': np.float32(.1)}),
                   dict(model__kernel_kws={'nested': [{'value': (np.float32(.1),)}]}),
                   dict(model__kernel_kws={'coef0': np.asarray(.1)}),
                   dict(model__theta=CustomFloat(.7)),
                   dict(model__kernel_kws=CustomDict(coef0=.1))]
        for parameters in changes:
            with self.subTest(parameters=parameters):
                self.assertIsNone(self.signature(pipeline().set_params(**parameters), labels))
        wrapped = pipeline(multiclass=True).set_params(model__estimator__theta=np.float32(.7))
        self.assertIsNone(self.signature(wrapped, np.repeat([0, 1, 2], 8)))

    def test_ovo_uses_every_pair_count_instead_of_full_fold_count(self):
        # Total counts coincide, but the smallest pair crosses a row boundary.
        labels = np.repeat([0, 1, 2], [4, 5, 7])
        a = pipeline('svm', multiclass=True).set_params(model__estimator__sv_selection_ratio=.55)
        b = clone(a).set_params(model__estimator__sv_selection_ratio=.56)
        self.assertEqual(int(16 * .55), int(16 * .56))
        self.assertNotEqual(int(9 * .55), int(9 * .56))
        self.assertNotEqual(self.signature(a, labels), self.signature(b, labels))
        # Conversely, a full-fold count difference is irrelevant to pair fits.
        labels = np.repeat([0, 1, 2], 4)
        a.set_params(model__estimator__sv_selection_ratio=.65)
        b.set_params(model__estimator__sv_selection_ratio=.70)
        self.assertNotEqual(int(12 * .65), int(12 * .70))
        self.assertEqual(int(8 * .65), int(8 * .70))
        self.assertEqual(self.signature(a, labels), self.signature(b, labels))

    def test_invalid_ratios_never_receive_equivalence_identity(self):
        for value in (0., -1., 1.1, True, float('nan'), float('inf')):
            with self.subTest(ratio=value):
                model = pipeline().set_params(model__sv_selection_ratio=value)
                self.assertIsNone(self.signature(model, np.arange(16) % 2))

    def test_custom_types_callables_and_unseeded_models_fall_back(self):
        labels = np.arange(16) % 2
        original = pipeline()
        custom_model = CustomDWD(**original.named_steps['model'].get_params())
        unsupported = [Pipeline([('scale', StandardScaler()), ('model', custom_model)]),
                       Pipeline([('scale', CustomScaler()), ('model', original.named_steps['model'])]),
                       clone(original).set_params(model__random_state=None),
                       clone(original).set_params(model__kernel=lambda X, Y: X @ Y.T),
                       clone(original).set_params(model__sv_selection_ratio=np.float64(.875))]
        for model in unsupported:
            with self.subTest(model=type(model.steps[-1][1]).__name__, params=str(model.get_params())[:120]):
                self.assertIsNone(self.signature(model, labels))

    def test_replacing_a_proven_builtin_method_disables_the_proof(self):
        original = EnsembleKernelDWD.fit
        def wrapped_fit(self, X, y):
            return original(self, X, y)
        with patch.object(EnsembleKernelDWD, 'fit', wrapped_fit):
            self.assertIsNone(self.signature(pipeline(), np.arange(16) % 2))


class EquivalentFoldReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.X, self.y = binary_data()
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.runtime_patch = patch.object(checkpoint, 'runtime_fingerprint',
                                         return_value={'test_runtime': 'fixed', 'native_threads': 1})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def session(self, name='run', *, estimator=None, resume=False, **kwargs):
        values = dict(run_dir=self.base / name, estimator=estimator if estimator is not None else pipeline(),
                      X=self.X, y=self.y, folds=self.folds, resume=resume,
                      reuse_equivalent_folds=True, provenance={'search': 'fixed-grid'})
        values.update(kwargs)
        (Path(values['run_dir']) / 'failures').mkdir(parents=True, exist_ok=True)
        return checkpoint.CheckpointedCV(**values)

    def receipts(self, name='run'):
        return [(path, json.loads(path.read_text())['payload'])
                for path in (self.base / name / 'cv-checkpoints' / 'folds').glob('*.json')]

    def assert_scores_equal(self, actual, expected):
        for name in ('params', 'mean_test_score', 'std_test_score', 'rank_test_score',
                     'split0_test_score', 'split1_test_score'):
            np.testing.assert_equal(actual.cv_results_[name], expected.cv_results_[name])
        self.assertEqual(actual.best_index_, expected.best_index_)
        self.assertEqual(actual.best_params_, expected.best_params_)
        self.assertEqual(actual.best_score_, expected.best_score_)
        self.assertEqual(actual.checkpoint_summary_['diagnostics'],
                         expected.checkpoint_summary_['diagnostics'])

    def test_actual_binary_dwd_and_svm_match_all_independent_fits(self):
        grid = {'model__sv_selection_ratio': [.875, .88]}
        for kind in ('dwd', 'svm'):
            with self.subTest(kind=kind):
                model = pipeline(kind)
                with self.session(kind + '-direct', estimator=model, reuse_equivalent_folds=False) as store:
                    expected = store.search(grid)
                with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit, \
                        self.session(kind, estimator=model) as store:
                    actual = store.search(grid)
                self.assertEqual(fit.call_count, 2)
                self.assert_scores_equal(actual, expected)
                self.assertEqual(actual.checkpoint_summary_['executed_folds'], 2)
                self.assertEqual(actual.checkpoint_summary_['equivalent_reused_folds'], 2)
                rows = self.receipts(kind)
                sources = {row['task_sha256']: row for _, row in rows if 'equivalent_reuse' not in row}
                aliases = [row for _, row in rows if 'equivalent_reuse' in row]
                self.assertEqual(len(sources), 2)
                self.assertEqual(len(aliases), 2)
                for row in aliases:
                    facts = row['equivalent_reuse']
                    source = sources[facts['source_task_sha256']]
                    self.assertEqual(facts['policy'], 'retained-count-v2')
                    self.assertEqual(facts['source_receipt_sha256'], checkpoint._digest(source))
                    self.assertEqual(facts['source_parameters'], source['parameters'])
                    self.assertEqual(facts['diagnostics_source_task_sha256'], source['task_sha256'])
                    self.assertEqual(facts['source_fit_time'], source['fit_time'])
                    self.assertEqual(facts['source_score_time'], source['score_time'])
                    self.assertGreaterEqual(facts['reuse_seconds'], 0.)
                    self.assertEqual(row['fit_time'], 0.)
                    self.assertEqual(row['score_time'], 0.)
                    self.assertEqual(row['diagnostics'], source['diagnostics'])
                    self.assertEqual(row['score'], source['score'])
                    self.assertNotEqual(row['task_sha256'], source['task_sha256'])

    def test_default_is_disabled_and_option_is_strict_boolean(self):
        args = dict(run_dir=self.base / 'default', estimator=pipeline('svm'),
                    X=self.X, y=self.y, folds=self.folds)
        with checkpoint.CheckpointedCV(**args) as store:
            result = store.search({'model__sv_selection_ratio': [.875, .88]})
        self.assertEqual(result.checkpoint_summary_['new_folds'], 4)
        self.assertNotIn('equivalent_reused_folds', result.checkpoint_summary_)
        self.assertTrue(all('equivalent_reuse' not in row for _, row in self.receipts('default')))
        for value in (None, 0, 1, 'yes', np.bool_(True)):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                checkpoint.CheckpointedCV(**args, reuse_equivalent_folds=value)

    def test_v1_checkpoint_and_source_cannot_resume_under_v2_policy(self):
        self.assertEqual(checkpoint._EQUIVALENT_POLICY, 'retained-count-v2')
        # Build a valid historical receipt without performing a numerical fit.
        with patch.object(checkpoint, '_EQUIVALENT_POLICY', 'retained-count-v1'), self.session() as store:
            parameters = {'model__sv_selection_ratio': .875}
            candidate = clone(store.estimator).set_params(**parameters)
            proof = checkpoint._fold_equivalence(candidate, store.identity_sha256,
                                                   0, store.y[store.folds[0][0]])
            task = checkpoint._digest({'identity': store.identity_sha256,
                                       'estimator': checkpoint._stable(candidate), 'fold': 0})
            record = {'schema': checkpoint.SCHEMA, 'identity_sha256': store.identity_sha256,
                      'task_sha256': task, 'parameters': checkpoint._stable(parameters),
                      'fold': 0, 'state': 'complete', 'score': .5, 'failure': None,
                      'fit_time': 0., 'score_time': 0., 'diagnostics': {},
                      'equivalent_fold': proof}
            source_path = store.folder / 'folds' / (task + '.json')
            manifest_path = store.folder / 'identity.json'
            checkpoint._write_receipt(source_path, record)
            self.assertEqual(proof['policy'], 'retained-count-v1')
            checkpoint._valid_fold(source_path, store.identity_sha256, task,
                                   parameters=parameters, fold=0)
            original_source = source_path.read_bytes()
            original_manifest = manifest_path.read_bytes()
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True):
            self.fail('A v1 checkpoint must not enter the v2 session')
        self.assertEqual(source_path.read_bytes(), original_source)
        self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_nearby_distinct_count_and_gamma_still_dispatch(self):
        grid = [{'model__sv_selection_ratio': [.875]},
                {'model__sv_selection_ratio': [.94]},
                {'model__sv_selection_ratio': [.88], 'model__gamma': [.3]}]
        with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit, self.session() as store:
            result = store.search(grid)
        self.assertEqual(fit.call_count, 6)
        self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 0)

    def test_numpy_theta_still_fits_normally_without_equivalent_reuse(self):
        X, y = self.X[:20], self.y[:20]
        folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))
        self.assertTrue(all(len(train) == 10 for train, _ in folds))
        model = pipeline('svm').set_params(model__theta=np.float32(.7))
        with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit, \
                self.session(estimator=model, X=X, y=y, folds=folds) as store:
            result = store.search({'model__sv_selection_ratio': [.8, .85]})
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(result.checkpoint_summary_['executed_folds'], 4)
        self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 0)
        self.assertTrue(np.isfinite(result.cv_results_['mean_test_score']).all())
        for candidate_index, ratio in enumerate((.8, .85)):
            for fold_index, (train, validation) in enumerate(folds):
                direct = clone(model).set_params(model__sv_selection_ratio=ratio).fit(X[train], y[train])
                direct_model = direct.named_steps['model']
                self.assertIs(type(direct_model.theta), np.float32)
                expected_subset_size = max(2, int(len(train) * direct_model.theta))
                self.assertEqual(len(direct_model.subset_indices_[0]), expected_subset_size)
                facts = result.checkpoint_summary_['diagnostics'][candidate_index][fold_index]
                self.assertEqual(facts['subset_sizes'], [expected_subset_size])
                self.assertEqual(facts['selected_size'], len(direct_model.selected_indices_))
                self.assertEqual(facts['selected_size'], 8)
                self.assertEqual(result.cv_results_[f'split{fold_index}_test_score'][candidate_index],
                                 direct.score(X[validation], y[validation]))
        self.assertTrue(all('equivalent_fold' not in row and 'equivalent_reuse' not in row
                            for _, row in self.receipts()))

    def test_invalid_ratio_is_validated_by_worker_and_records_failure(self):
        with self.assertRaisesRegex(ValueError, 'sv_selection_ratio'), \
                patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit, \
                self.session(estimator=pipeline('svm')) as store:
            store.search({'model__sv_selection_ratio': [1.01]})
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(self.receipts(), [])
        attempts = [json.loads(path.read_text())['payload'] for path in
                    (self.base / 'run' / 'cv-checkpoints' / 'attempts').glob('*.json')]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['state'], 'failed')
        self.assertNotIn('equivalent_reuse', attempts[0])
        failures = list((self.base / 'run' / 'failures').glob('failure-*.json'))
        self.assertEqual(len(failures), 1)
        failure = json.loads(failures[0].read_text())
        self.assertEqual(failure['phase'], 'cv_fit')
        self.assertEqual(failure['exception_type'], 'builtins.ValueError')

    def test_actual_ovo_pair_fits_match_independent_candidates(self):
        X, y = multiclass_data()
        folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))
        model = pipeline('svm', multiclass=True)
        grid = {'model__estimator__sv_selection_ratio': [.65, .70]}
        options = dict(estimator=model, X=X, y=y, folds=folds)
        with self.session('ovo-direct', reuse_equivalent_folds=False, **options) as store:
            expected = store.search(grid)
        with self.session('ovo', **options) as store:
            actual = store.search(grid)
        self.assert_scores_equal(actual, expected)
        self.assertEqual(actual.checkpoint_summary_['executed_folds'], 2)
        self.assertEqual(actual.checkpoint_summary_['equivalent_reused_folds'], 2)
        for fold in actual.checkpoint_summary_['diagnostics'][1]:
            pairs = fold['pairwise_estimators']
            self.assertEqual(len(pairs), 3)
            self.assertTrue(all(pair['diagnostics']['selected_size'] == 5 for pair in pairs))

    def test_reuse_across_stages_has_no_alias_chains_and_resume_runs_no_fits(self):
        with self.session(estimator=pipeline('svm')) as store:
            store.search({'model__sv_selection_ratio': [.875]}, stage='initial')
            store.search({'model__sv_selection_ratio': [.88]}, stage='refine-1')
            third = store.search({'model__sv_selection_ratio': [.89]}, stage='refine-2')
        sources = {row['task_sha256'] for _, row in self.receipts() if 'equivalent_reuse' not in row}
        self.assertEqual(len(sources), 2)
        for _, row in self.receipts():
            if 'equivalent_reuse' in row:
                self.assertIn(row['equivalent_reuse']['source_task_sha256'], sources)
        with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('Completed aliases must not fit')), \
                self.session(estimator=pipeline('svm'), resume=True) as store:
            resumed = store.search({'model__sv_selection_ratio': [.89]})
        self.assert_scores_equal(resumed, third)
        self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)
        self.assertEqual(resumed.checkpoint_summary_['executed_folds'], 0)
        self.assertEqual(resumed.checkpoint_summary_['equivalent_reused_folds'], 2)

    def test_completed_source_survives_interruption_before_alias_commit(self):
        real_write = checkpoint._write_receipt
        def interrupt_alias(path, payload):
            if 'equivalent_reuse' in payload:
                raise KeyboardInterrupt('Interrupted before alias publication')
            return real_write(path, payload)
        with self.assertRaises(KeyboardInterrupt), self.session(estimator=pipeline('svm')) as store:
            store.search({'model__sv_selection_ratio': [.875]}, stage='initial')
            with patch.object(checkpoint, '_write_receipt', side_effect=interrupt_alias):
                store.search({'model__sv_selection_ratio': [.88]}, stage='refine')
        self.assertEqual(len(self.receipts()), 2)
        self.assertTrue(all('equivalent_reuse' not in row for _, row in self.receipts()))
        with patch.object(checkpoint, '_fit_fold', side_effect=AssertionError('Durable sources must survive')), \
                self.session(estimator=pipeline('svm'), resume=True) as store:
            result = store.search({'model__sv_selection_ratio': [.875, .88]})
        self.assertEqual(result.checkpoint_summary_['executed_folds'], 0)
        self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 2)

    def test_missing_or_corrupt_alias_source_is_rejected(self):
        for damage in ('missing', 'checksum', 'changed-valid-checksum'):
            with self.subTest(damage=damage):
                with self.session(damage, estimator=pipeline('svm')) as store:
                    store.search({'model__sv_selection_ratio': [.875, .88]})
                sources = [(p, row) for p, row in self.receipts(damage) if 'equivalent_reuse' not in row]
                path, row = sources[0]
                if damage == 'missing':
                    path.unlink()
                elif damage == 'checksum':
                    envelope = json.loads(path.read_text())
                    envelope['payload']['score'] = .123
                    path.write_text(json.dumps(envelope))
                else:
                    row['score'] = .123
                    checkpoint._write_receipt(path, row)
                with self.assertRaises(checkpoint.CheckpointError), \
                        self.session(damage, estimator=pipeline('svm'), resume=True) as store:
                    store.search({'model__sv_selection_ratio': [.88]})

    def test_ineligible_source_is_not_an_equivalence_representative(self):
        with patch.object(checkpoint, '_fit_fold', wraps=checkpoint._fit_fold) as fit, \
                self.session(estimator=pipeline('svm')) as store:
            result = store.search({'model__sv_selection_ratio': [.01, .02]})
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(result.checkpoint_summary_['executed_folds'], 4)
        self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 0)
        self.assertTrue(np.isnan(result.cv_results_['mean_test_score']).all())
        self.assertTrue(all(row['state'] == 'ineligible' and 'equivalent_reuse' not in row
                            for _, row in self.receipts()))

    def test_failed_attempt_cannot_supply_an_alias_and_is_retried(self):
        with self.assertRaises(FloatingPointError), \
                patch.object(checkpoint.CandidateEvaluator, 'fit', side_effect=FloatingPointError('tiny failure')), \
                self.session(estimator=pipeline('svm')) as store:
            store.search({'model__sv_selection_ratio': [.875, .88]})
        self.assertEqual(self.receipts(), [])
        attempts = self.base / 'run' / 'cv-checkpoints' / 'attempts'
        self.assertIn('failed', [json.loads(path.read_text())['payload']['state'] for path in attempts.glob('*.json')])
        with self.session(estimator=pipeline('svm'), resume=True) as store:
            result = store.search({'model__sv_selection_ratio': [.875, .88]})
        self.assertEqual(result.checkpoint_summary_['executed_folds'], 2)
        self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 2)


class ParallelEquivalentFoldReuseTests(unittest.TestCase):
    def test_two_process_dispatch_commits_sources_before_aliases(self):
        # A real runtime identity is required: monkeypatches do not transfer to loky workers.
        X, y = binary_data()
        folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))
        with tempfile.TemporaryDirectory() as temporary:
            settings = dict(run_dir=Path(temporary), estimator=pipeline('svm'), X=X, y=y,
                            folds=folds, jobs=2, reuse_equivalent_folds=True)
            with checkpoint.CheckpointedCV(**settings) as store:
                result = store.search({'model__sv_selection_ratio': [.875, .88]})
            self.assertEqual(result.checkpoint_summary_['executed_folds'], 2)
            self.assertEqual(result.checkpoint_summary_['equivalent_reused_folds'], 2)
            self.assertTrue(np.isfinite(result.cv_results_['mean_test_score']).all())
            rows = [json.loads(path.read_text())['payload']
                    for path in (Path(temporary) / 'cv-checkpoints' / 'folds').glob('*.json')]
            source_pids = {row['pid'] for row in rows if 'equivalent_reuse' not in row}
            import os
            self.assertNotIn(os.getpid(), source_pids)
            with checkpoint.CheckpointedCV(**settings, resume=True) as store:
                resumed = store.search({'model__sv_selection_ratio': [.875, .88]})
            self.assertEqual(resumed.checkpoint_summary_['executed_folds'], 0)
            self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 4)
            self.assertEqual(resumed.checkpoint_summary_['equivalent_reused_folds'], 2)
            np.testing.assert_equal(resumed.cv_results_['mean_test_score'], result.cv_results_['mean_test_score'])


if __name__ == '__main__':
    unittest.main()
