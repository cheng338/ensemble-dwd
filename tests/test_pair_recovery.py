"""Trusted local pair recovery preserves fits and checkpoint ownership."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, OvOClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
import pair_recovery as recovery
import _pair_store as storage
from work_reporting import aggregate_work_summaries, summarize_records


def read(path):
    return json.loads(checkpoint._io_path(path).read_text(encoding='utf-8'))


def hashes(folder, pattern):
    folder = checkpoint._io_path(folder)
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob(pattern)}


def pipeline(mode='full'):
    controls = dict(lambd=.001, max_iter=20, obj_tol=1e-7, tol=1e-6,
                    random_state=42, prediction_batch_size=32)
    if mode == 'full':
        binary = KernGDWD(kernel='rbf', kernel_kws={'gamma': 1 / 6},
                          initialization='zero', **controls)
    else:
        binary = EnsembleKernelDWD(M=2, theta=.4, sv_selection_ratio=.75,
                    gamma=1 / 6, store_base_estimators=mode == 'retained', **controls)
    return Pipeline([('scale', StandardScaler()), ('model', OvOClassifier(binary, n_jobs=1))])


class PairRecoveryIntegrationTests(unittest.TestCase):
    """Small real DWD fits; runtime identity is measured rather than mocked."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pair-cv-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.X, self.y = make_blobs(n_samples=96, n_features=6, centers=4,
                                   cluster_std=2., random_state=20261007)
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()
        self.addCleanup(self.thread_limit.__exit__, None, None, None)

    def session(self, name, *, mode='full', resume=False, enabled=True, **kwargs):
        folder = self.root / name
        (folder / 'failures').mkdir(parents=True, exist_ok=True)
        return checkpoint.CheckpointedCV(folder, pipeline(mode), self.X, self.y,
                    self.folds, jobs=1, error_policy='raise', resume=resume,
                    provenance={'test': 'pair-recovery-v1'}, pair_recovery=enabled, **kwargs)

    def execute(self, name, **kwargs):
        with self.session(name, **kwargs) as session:
            return session.search({})

    @contextmanager
    def interrupt_after_two_pairs(self):
        original = recovery.FoldContext._pair_finished

        def finish(context, event):
            original(context, event)
            if context.fold == 1 and len(context.events) == 2:
                raise KeyboardInterrupt('controlled pair boundary')

        with patch.object(recovery.FoldContext, '_pair_finished', finish):
            yield

    @contextmanager
    def observe_models(self, collection):
        original = checkpoint.eligible_accuracy

        def score(evaluator, X, y):
            result = original(evaluator, X, y)
            if evaluator.eligible_:
                fitted = evaluator.estimator_
                scaler, ovo = (step for _, step in fitted.steps)
                transformed = scaler.transform(self.X)
                collection.append({
                    'scaler': recovery.snapshot(scaler),
                    'pairs': recovery.snapshot(ovo.pairs_),
                    'rows': recovery.snapshot(ovo.pair_indices_),
                    'models': recovery.snapshot(ovo.estimators_),
                    'scores': [child.decision_function(transformed).copy() for child in ovo.estimators_],
                    'votes': ovo.vote_counts(transformed),
                    'predictions': fitted.predict(self.X),
                })
            return result

        with patch.object(checkpoint, 'eligible_accuracy', score):
            yield

    def test_real_full_lean_and_retained_models_match_after_partial_resume(self):
        for mode in ('full', 'lean', 'retained'):
            with self.subTest(mode=mode):
                baseline, restored = [], []
                with self.observe_models(baseline):
                    expected = self.execute(mode + '-plain', mode=mode, enabled=False)
                with self.observe_models(restored), self.interrupt_after_two_pairs(), \
                        self.assertRaisesRegex(KeyboardInterrupt, 'controlled pair boundary'):
                    self.execute(mode, mode=mode)
                before = hashes(self.root / mode / 'cv-checkpoints', '*.pkl')
                self.assertEqual(len(before), 2)
                completed_before = hashes(self.root / mode / 'cv-checkpoints/folds', '*.json')
                self.assertEqual(len(completed_before), 1)
                with self.observe_models(restored):
                    actual = self.execute(mode, mode=mode, resume=True)
                self.assertEqual(len(restored), 2)
                for expected_model, actual_model in zip(baseline, restored):
                    for name in ('scaler', 'pairs', 'rows', 'models'):
                        self.assertEqual(expected_model[name], actual_model[name])
                    for expected_scores, actual_scores in zip(expected_model['scores'], actual_model['scores']):
                        np.testing.assert_array_equal(expected_scores, actual_scores)
                    np.testing.assert_array_equal(expected_model['votes'], actual_model['votes'])
                    np.testing.assert_array_equal(expected_model['predictions'], actual_model['predictions'])
                for name in ('params', 'mean_test_score', 'std_test_score', 'rank_test_score',
                             'split0_test_score', 'split1_test_score'):
                    np.testing.assert_equal(expected.cv_results_[name], actual.cv_results_[name])
                after = hashes(self.root / mode / 'cv-checkpoints/folds', '*.json')
                self.assertTrue(all(after[name] == value for name, value in completed_before.items()))
                self.assertEqual(hashes(self.root / mode / 'cv-checkpoints', '*.pkl'), {})
                work = actual.work_summary_['pair_recovery']['current']
                self.assertEqual(work['fitted_pairs'], 4)
                self.assertEqual(work['recovered_pairs'], 2)
                self.assertEqual(work['fitted_unsaved_pairs'], 0)
                with patch.object(recovery, 'fit_pipeline', side_effect=AssertionError('completed folds refitted')):
                    replay = self.execute(mode, mode=mode, resume=True)
                self.assertEqual(replay.checkpoint_summary_['reused_folds'], 2)
                self.assertEqual(after, hashes(self.root / mode / 'cv-checkpoints/folds', '*.json'))

    def test_disabled_path_never_creates_recovery_context_or_artifacts(self):
        with patch.object(recovery, 'FoldContext', side_effect=AssertionError('recovery entered')):
            result = self.execute('disabled', enabled=False)
        self.assertNotIn('pair_recovery', result.work_summary_)
        folder = self.root / 'disabled/cv-checkpoints'
        self.assertFalse((folder / 'pair-stores').exists())
        self.assertNotIn('pair_recovery', read(folder / 'identity.json')['payload']['identity'])
        for path in (folder / 'folds').glob('*.json'):
            self.assertNotIn('pair_recovery', read(path)['payload'])

    def test_unsupported_combinations_reject_before_fitting(self):
        for options in ({'reuse_equivalent_folds': True},
                        {'cv_seeds': [42, 43], 'seed_parameter': 'model__estimator__random_state'},
                        {'pair_recovery_max_bytes': -1},
                        {'pair_recovery_max_bytes': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.session('unsupported', **options)
        with self.assertRaises(ValueError):
            self.session('unused-quota', enabled=False, pair_recovery_max_bytes=100)

    def test_unsupported_estimator_contracts_reject_explicitly(self):
        cases = []
        model = pipeline()
        cases.append(model.set_params(model__n_jobs=2))
        model = pipeline()
        cases.append(model.set_params(scale=MinMaxScaler()))
        model = pipeline()
        cases.append(model.set_params(model__estimator__kernel='linear'))
        model = pipeline()
        cases.append(model.set_params(model__estimator__random_state=np.random.RandomState(42)))
        model = pipeline()
        cases.append(model.set_params(model__estimator__kernel_kws={'gamma': 'scale'}))
        model = pipeline('lean')
        cases.append(model.set_params(model__estimator__cache_dir=str(self.root / 'cache')))
        cases.append(pipeline().set_params(memory=str(self.root / 'pipeline-cache')))
        cases.append(pipeline().set_params(verbose=True))
        for model in cases:
            with self.subTest(model=model), self.assertRaises((ValueError, TypeError)):
                recovery.validate_supported(model)


class PairStoreTests(unittest.TestCase):
    """Faults use plain dictionaries; numerical model tests are separate."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pair-store-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.active = True
        self.key = {'rows': [2, 4], 'labels': [-1, 1], 'settings': {'seed': 42}}
        self.model = {'coef': [1.25, -.5], 'intercept': 2.}

    def owner(self):
        if not self.active:
            raise checkpoint.CheckpointBusyError('stale test generation')

    def store(self, name='pair', max_bytes=None, identity=None):
        return storage.PairStore(self.root / name, {'fold': 1} if identity is None else identity,
                                 max_bytes=max_bytes, snapshot=deepcopy, assert_owner=self.owner)

    def seed(self, store):
        with store.session() as session:
            model, event = session.get_or_compute(self.key, lambda: deepcopy(self.model), lambda value: None)
        self.assertEqual(model, self.model)
        self.assertEqual(event['status'], 'fitted_saved')
        return event

    def test_restoration_has_no_fit_and_preserves_prior_cost_separately(self):
        store = self.store()
        first = self.seed(store)
        compute = Mock(side_effect=AssertionError('restored pair refitted'))
        validate = Mock()
        with store.session() as session:
            model, event = session.get_or_compute(self.key, compute, validate)
        self.assertEqual(model, self.model)
        compute.assert_not_called()
        validate.assert_called_once_with(model)
        self.assertEqual(event['status'], 'reused')
        self.assertEqual(event['fit_seconds'], 0.)
        self.assertEqual(event['save_seconds'], 0.)
        self.assertEqual(event['historical_fit_seconds'], first['fit_seconds'])
        self.assertGreaterEqual(event['load_seconds'], 0.)

    def test_identity_checksum_size_and_filename_reject_before_unpickling(self):
        variants = ('parent', 'key', 'filename', 'payload', 'size', 'envelope')
        for variant in variants:
            with self.subTest(variant=variant):
                store = self.store(variant)
                event = self.seed(store)
                receipt = storage._io_path(store.root / (event['key'] + '.json'))
                payload = read(receipt)['payload']
                model_path = storage._io_path(store.root / (event['key'] + '.pkl'))
                if variant == 'parent':
                    payload['parent_sha256'] = '0' * 64
                elif variant == 'key':
                    payload['key_identity']['rows'].reverse()
                elif variant == 'filename':
                    payload['model_file'] = '../other.pkl'
                elif variant == 'payload':
                    data = bytearray(model_path.read_bytes())
                    data[-1] ^= 1
                    model_path.write_bytes(data)
                elif variant == 'size':
                    payload['model_bytes'] += 1
                receipt.write_bytes(storage._envelope(payload))
                if variant == 'envelope':
                    envelope = read(receipt)
                    envelope['sha256'] = '0' * 64
                    receipt.write_text(json.dumps(envelope), encoding='utf-8')
                with store.session() as session, \
                        patch.object(storage.pickle, 'loads', side_effect=AssertionError('unverified unpickle')) as loads, \
                        self.assertRaises(storage.StoreMismatch):
                    session.get_or_compute(self.key, Mock(side_effect=AssertionError('refit')), lambda value: None)
                loads.assert_not_called()

    def test_parent_mismatch_preserves_committed_artifacts(self):
        original = self.store()
        self.seed(original)
        before = hashes(original.root, '*')
        with self.assertRaises(storage.StoreMismatch), self.store(identity={'fold': 2}).session():
            pass
        self.assertEqual(before, hashes(original.root, '*'))

    def test_quota_and_write_failure_preserve_valid_fresh_model(self):
        for failure in ('quota', 'parent', 'payload', 'manifest', 'serialization'):
            with self.subTest(failure=failure):
                store = self.store(failure, max_bytes=0 if failure == 'quota' else None)
                original = store._atomic_write

                def write(name, data):
                    should_fail = ((failure == 'parent' and name == 'identity.json') or
                                   (failure == 'payload' and name.endswith('.pkl')) or
                                   (failure == 'manifest' and name.endswith('.json') and name != 'identity.json'))
                    if should_fail:
                        raise OSError('controlled cache write failure')
                    return original(name, data)

                with patch.object(store, '_atomic_write', write), store.session() as session:
                    if failure == 'serialization':
                        with patch.object(storage.pickle, 'dumps', side_effect=OSError('controlled serialization failure')):
                            model, event = session.get_or_compute(self.key, lambda: deepcopy(self.model), lambda value: None)
                    else:
                        model, event = session.get_or_compute(self.key, lambda: deepcopy(self.model), lambda value: None)
                self.assertEqual(model, self.model)
                self.assertEqual(event['status'], 'fitted_unsaved')
                self.assertIs(event['cache_committed'], False)
                self.assertEqual(len(list(storage._io_path(store.root).glob('*.json'))),
                                 0 if failure == 'parent' else 1)

    def test_compute_and_validation_errors_are_not_optional_cache_failures(self):
        for phase in ('compute', 'validate'):
            error = FloatingPointError('original numerical failure')
            compute = Mock(side_effect=error) if phase == 'compute' else lambda: deepcopy(self.model)
            validate = Mock(side_effect=error) if phase == 'validate' else lambda value: None
            with self.subTest(phase=phase), self.store(phase).session() as session, \
                    self.assertRaises(FloatingPointError) as caught:
                session.get_or_compute(self.key, compute, validate)
            self.assertIs(caught.exception, error)
            self.assertEqual(hashes(self.root / phase, '*.pkl'), {})

    def test_generation_is_fenced_before_work_and_after_computation(self):
        store = self.store()
        with store.session() as session:
            self.active = False
            compute = Mock(side_effect=AssertionError('stale owner computed'))
            with self.assertRaises(checkpoint.CheckpointBusyError):
                session.get_or_compute(self.key, compute, lambda value: None)
            compute.assert_not_called()
        self.active = True
        with store.session() as session:
            def compute():
                self.active = False
                return deepcopy(self.model)
            with self.assertRaises(checkpoint.CheckpointBusyError):
                session.get_or_compute(self.key, compute, lambda value: None)
        self.assertEqual(hashes(store.root, '*.pkl'), {})

    def test_committed_corruption_is_preserved_but_uncommitted_payload_is_reclaimed(self):
        store = self.store()
        event = self.seed(store)
        root = storage._io_path(store.root)
        (root / (event['key'] + '.json')).unlink()
        orphan = event['key'] + '.pkl'
        with store.session() as session:
            self.assertIn(orphan, session.reclaimed_files)
            self.assertFalse((root / orphan).exists())
        event = self.seed(store)
        path = root / (event['key'] + '.json')
        path.write_text('invalid committed receipt', encoding='utf-8')
        before = hashes(root, '*')
        with store.session() as session, self.assertRaises(storage.StoreMismatch):
            session.get_or_compute(self.key, Mock(side_effect=AssertionError('refit')), lambda value: None)
        self.assertEqual(before, hashes(root, '*'))

    def test_cleanup_retains_parent_and_never_reopens_completed_session(self):
        store = self.store()
        self.seed(store)
        root = storage._io_path(store.root)
        parent = (root / 'identity.json').read_bytes()
        with store.session() as session:
            event = session.complete('a' * 64)
            self.assertEqual(event['status'], 'cleaned')
            self.assertEqual(event['remaining_pair_bytes'], 0)
            self.assertEqual((root / 'identity.json').read_bytes(), parent)
            with self.assertRaises(RuntimeError):
                session.get_or_compute(self.key, lambda: self.model, lambda value: None)
        self.assertEqual(hashes(root, '*.pkl'), {})


class ProtocolBinary(ClassifierMixin, BaseEstimator):
    """A labeled centroid stub for control-flow faults, with no DWD solver."""
    fits = 0
    failure_at = None
    failure = None

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        type(self).fits += 1
        if type(self).fits == type(self).failure_at:
            raise type(self).failure
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        positive, negative = X[y == 1].mean(axis=0), X[y == -1].mean(axis=0)
        self.coef_ = positive - negative
        self.intercept_ = -float((positive + negative) @ self.coef_) / 2
        self.cache_hit_, self.cache_status_ = False, 'disabled'
        return self

    def predict(self, X):
        return np.where(X @ self.coef_ + self.intercept_ > 0, 1, -1)


class PairCoordinatorProtocolTests(unittest.TestCase):
    """Actual CV leases, tokens, runtime and receipts; only model is a stub."""
    setUp = PairRecoveryIntegrationTests.setUp

    def session(self, *, resume=False, error_policy='raise'):
        folder = self.root / 'run'
        (folder / 'failures').mkdir(parents=True, exist_ok=True)
        model = Pipeline([('scale', StandardScaler()),
                          ('model', OvOClassifier(ProtocolBinary(), n_jobs=1))])
        return checkpoint.CheckpointedCV(folder, model, self.X, self.y,
                self.folds, jobs=1, error_policy=error_policy, resume=resume,
                provenance={'test': 'centroid-protocol-only'}, pair_recovery=True)

    @contextmanager
    def contract(self):
        ProtocolBinary.fits = 0
        ProtocolBinary.failure_at = ProtocolBinary.failure = None

        def validate(model, estimator, X, y):
            self.assertIs(type(model), ProtocolBinary)
            self.assertEqual(model.n_features_in_, X.shape[1])
            np.testing.assert_array_equal(model.classes_, np.unique(y))
            self.assertTrue(np.isfinite(model.coef_).all())

        with patch.object(recovery, 'validate_supported', lambda *args, **kwargs: None), \
                patch.object(recovery, 'validate_model', validate):
            yield

    def execute(self, **kwargs):
        with self.session(**kwargs) as session:
            return session.search({})

    @property
    def folder(self):
        return checkpoint._io_path(self.root / 'run/cv-checkpoints')

    def test_partial_ineligible_fit_keeps_original_nan_policy_and_work(self):
        with self.contract():
            ProtocolBinary.failure_at = 2
            ProtocolBinary.failure = IneligibleSelectionError('controlled ineligible pair')
            result = self.execute(error_policy='nan')
            self.assertTrue(np.isnan(result.cv_results_['mean_test_score'][0]))
            records = [read(path)['payload'] for path in (self.folder / 'folds').glob('*.json')]
            ineligible = next(record for record in records if record['state'] == 'ineligible')
            self.assertEqual(ineligible['pair_recovery']['fitted_pairs'], 1)
            self.assertEqual(ineligible['failure']['category'], 'ineligible_selection')
            self.assertEqual(hashes(self.folder, '*.pkl'), {})
            before = ProtocolBinary.fits
            resumed = self.execute(error_policy='nan', resume=True)
            self.assertEqual(ProtocolBinary.fits, before)
            self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)

    def test_reporting_failure_does_not_mask_original_fit_error(self):
        original = checkpoint._write_receipt

        def write(path, payload):
            if 'pair-evidence' in Path(path).parts:
                raise OSError('controlled evidence write failure')
            return original(path, payload)

        with self.contract(), patch.object(checkpoint, '_write_receipt', write):
            error = FloatingPointError('original solver failure')
            ProtocolBinary.failure_at, ProtocolBinary.failure = 2, error
            with self.assertRaises(FloatingPointError) as caught:
                self.execute()
            self.assertIs(caught.exception, error)
        attempts = [read(path)['payload'] for path in (self.folder / 'attempts').glob('*.json')]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['state'], 'failed')
        self.assertEqual(attempts[0]['failure']['message'], 'original solver failure')
        self.assertEqual(len(hashes(self.folder, '*.pkl')), 1)

    def test_after_commit_interrupt_reuses_authoritative_fold_and_prunes(self):
        original = recovery.cleanup_context

        def cleanup(context, **kwargs):
            if context.fold == 1:
                raise KeyboardInterrupt('controlled post-commit interruption')
            return original(context, **kwargs)

        with self.contract():
            with patch.object(recovery, 'cleanup_context', cleanup), self.assertRaises(KeyboardInterrupt):
                self.execute()
            before = hashes(self.folder / 'folds', '*.json')
            self.assertEqual(len(before), 2)
            self.assertEqual(len(hashes(self.folder, '*.pkl')), 6)
            fits = ProtocolBinary.fits
            with patch.object(storage.pickle, 'loads', side_effect=AssertionError('completed fold loaded pairs')):
                resumed = self.execute(resume=True)
            self.assertEqual(ProtocolBinary.fits, fits)
            self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)
            self.assertEqual(before, hashes(self.folder / 'folds', '*.json'))
            self.assertEqual(hashes(self.folder, '*.pkl'), {})
            interrupted = [read(path)['payload'] for path in (self.folder / 'attempts').glob('*.json')
                           if read(path)['payload']['state'] == 'interrupted']
            self.assertEqual(len(interrupted), 1)
            self.assertIs(interrupted[0]['fold_receipt_committed'], True)

    def test_invalid_authoritative_receipt_never_authorizes_cleanup_or_refit(self):
        with self.contract(), patch.object(recovery, '_cleanup_store', side_effect=OSError('controlled cleanup failure')):
            self.execute()
        saved = hashes(self.folder, '*.pkl')
        self.assertEqual(len(saved), 12)
        path = next(path for path in (self.folder / 'folds').glob('*.json')
                    if read(path)['payload']['fold'] == 0)
        original = path.read_bytes()
        variants = ('identity', 'task', 'parameters', 'counts', 'state', 'checksum',
                    'recovery_count', 'recovery_timer', 'recovery_origin')
        for variant in variants:
            envelope = json.loads(original)
            record = envelope['payload']
            if variant == 'identity':
                record['identity_sha256'] = '0' * 64
            elif variant == 'task':
                record['task_sha256'] = '0' * 64
            elif variant == 'parameters':
                record['parameters'] = checkpoint._stable({'wrong': 1})
            elif variant == 'counts':
                record['accuracy_counts']['n_validation'] += 1
            elif variant == 'state':
                record['state'] = 'running'
            elif variant == 'recovery_count':
                record['pair_recovery']['fitted_pairs'] += 1
            elif variant == 'recovery_timer':
                record['pair_recovery']['pair_load_seconds'] += 1.
            elif variant == 'recovery_origin':
                record['pair_recovery']['pairs'][0]['work_origin'] = 'recovered'
            envelope['sha256'] = '0' * 64 if variant == 'checksum' else checkpoint._digest(record)
            path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.subTest(variant=variant), self.contract(), \
                    patch.object(recovery, '_cleanup_store', side_effect=AssertionError('invalid cleanup')), \
                    self.assertRaises(checkpoint.CheckpointError):
                self.execute(resume=True)
            self.assertEqual(ProtocolBinary.fits, 0)
            self.assertEqual(saved, hashes(self.folder, '*.pkl'))
        path.write_bytes(original)

    def test_saved_fold_cleanup_holds_existing_task_lease(self):
        with self.contract(), patch.object(recovery, '_cleanup_store', side_effect=OSError('controlled cleanup failure')):
            self.execute()
        original = recovery._cleanup_store
        observed = []

        def cleanup(context, record):
            lock = context.folder / 'locks' / (context.task + '.lock')
            with self.assertRaises(checkpoint.CheckpointBusyError), checkpoint._Lease(lock):
                pass
            observed.append(context.fold)
            return original(context, record)

        with self.contract(), patch.object(recovery, '_cleanup_store', cleanup):
            resumed = self.execute(resume=True)
            self.assertEqual(ProtocolBinary.fits, 0)
        self.assertEqual(sorted(observed), [0, 1])
        self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 2)
        self.assertEqual(hashes(self.folder, '*.pkl'), {})

    def test_changed_generation_does_not_publish_unfinished_fold(self):
        original = recovery.FoldContext._pair_finished

        def finish(context, event):
            original(context, event)
            if context.fold == 1 and len(context.events) == 6:
                checkpoint.write_json_atomic(context.folder / 'owner.json',
                    {'token': 'new-generation', 'state': 'active'}, replace=True)

        with self.contract():
            with patch.object(recovery.FoldContext, '_pair_finished', finish), \
                    self.assertRaises(checkpoint.CheckpointBusyError):
                self.execute()
            self.assertEqual(len(hashes(self.folder / 'folds', '*.json')), 1)
            self.assertEqual(len(hashes(self.folder, '*.pkl')), 6)
            before = ProtocolBinary.fits
            resumed = self.execute(resume=True)
            self.assertEqual(ProtocolBinary.fits, before)
            self.assertEqual(resumed.work_summary_['pair_recovery']['current']['recovered_pairs'], 6)
            self.assertEqual(resumed.work_summary_['base_cache']['reported_disabled'], 0)
            self.assertEqual(resumed.work_summary_['base_cache']['reported_hits'], 0)
            self.assertEqual(resumed.work_summary_['base_cache']['unknown_diagnostic_nodes'], 0)
            self.assertEqual(resumed.work_summary_['pair_recovery']['current']['fitted_pairs'], 0)

    def test_source_change_rejects_resume_before_pair_loading(self):
        with self.contract(), patch.object(recovery, '_cleanup_store', side_effect=OSError('controlled cleanup failure')):
            self.execute()
        original = checkpoint._file_hash

        def changed(path):
            return 'a' * 64 if Path(path).name == '_pair_store.py' else original(path)

        before = hashes(self.folder, '*.pkl')
        with self.contract(), patch.object(checkpoint, '_file_hash', changed), \
                patch.object(storage.pickle, 'loads', side_effect=AssertionError('changed source loaded pair')), \
                self.assertRaises(checkpoint.CheckpointError):
            self.execute(resume=True)
        self.assertEqual(before, hashes(self.folder, '*.pkl'))
        self.assertEqual(ProtocolBinary.fits, 0)


class PairRecoveryReportingTests(unittest.TestCase):
    def test_current_pair_cache_observations_exclude_recovered_history(self):
        recovery_record = {'schema': 1, 'fitted_pairs': 1, 'recovered_pairs': 1,
            'fitted_unsaved_pairs': 0, 'current_pair_fit_seconds': .5,
            'pair_load_seconds': .02, 'pair_save_seconds': .01,
            'historical_reused_pair_fit_seconds': 900., 'cleanup_status': 'cleaned'}
        record = {'state': 'complete', 'fit_time': .75, 'score_time': .1,
                  'pair_recovery': recovery_record, 'diagnostics': {'pairwise_estimators': [
                      {'pair': [0, 1], 'work_origin': 'recovered',
                       'diagnostics': {'cache_hit_': True, 'cache_status_': 'hit'}},
                      {'pair': [0, 2], 'work_origin': 'fitted',
                       'diagnostics': {'cache_hit_': False, 'cache_status_': 'disabled'}}]}}
        original = deepcopy(record)
        summary = summarize_records([(record, False)])
        self.assertEqual(summary['current_worker_seconds']['fit'], .75)
        self.assertEqual(summary['base_cache']['reported_hits'], 0)
        self.assertEqual(summary['base_cache']['reported_disabled'], 1)
        group = summary['pair_recovery']
        self.assertEqual(group['current']['recovered_pairs'], 1)
        self.assertEqual(group['current_worker_seconds'], {'pair_fit': .5, 'pair_load': .02, 'pair_save': .01})
        self.assertEqual(group['historical_reused_pair_fit_seconds'], 900.)
        replayed = summarize_records([(record, True)])
        self.assertEqual(replayed['pair_recovery']['current']['recovered_pairs'], 0)
        self.assertEqual(replayed['pair_recovery']['replayed_receipts']['recovered_pairs'], 1)
        self.assertEqual(replayed['current_worker_seconds']['fit'], 0.)
        aggregated = aggregate_work_summaries([summary, replayed])
        self.assertEqual(aggregated['pair_recovery']['current']['recovered_pairs'], 1)
        self.assertEqual(aggregated['pair_recovery']['replayed_receipts']['recovered_pairs'], 1)
        self.assertEqual(record, original)

    def test_mixed_disabled_and_recovery_summaries_do_not_invent_pair_work(self):
        plain = {'state': 'complete', 'fit_time': 2., 'score_time': .1,
                 'diagnostics': {'cache_hit_': False, 'cache_status_': 'disabled'}}
        plain_summary = summarize_records([(plain, False)])
        self.assertNotIn('pair_recovery', plain_summary)
        resumed = {**plain, 'pair_recovery': {'schema': 1, 'fitted_pairs': 0,
            'recovered_pairs': 1, 'fitted_unsaved_pairs': 0,
            'current_pair_fit_seconds': 0., 'pair_load_seconds': .02,
            'pair_save_seconds': 0., 'historical_reused_pair_fit_seconds': 8.,
            'cleanup_status': 'cleaned'}}
        optional_summary = summarize_records([(resumed, False)])
        combined = aggregate_work_summaries([plain_summary, optional_summary])
        self.assertEqual(combined['pair_recovery'], optional_summary['pair_recovery'])
        self.assertEqual(combined['current_worker_seconds']['fit'], 4.)


class PairRecoveryCliTests(unittest.TestCase):
    def test_confirmation_helper_rejects_invalid_options_before_any_work(self):
        from numerical_confirmation import run_numerical_confirmation
        options = dict(pipeline=None, results=None, rows=None, searches=None,
            folds=None, X=None, y=None, run_dir=None, config=None, search_config=None,
            jobs=1, error_policy='raise', resume=False, cv_seeds=None,
            seed_parameter=None, reuse_equivalent_folds=False, parent_identity=None)
        cases = ({'pair_recovery': False, 'pair_recovery_max_bytes': 0},
                 {'pair_recovery': True, 'pair_recovery_max_bytes': -1},
                 {'pair_recovery': True, 'pair_recovery_max_bytes': True},
                 {'pair_recovery': 1},
                 {'pair_recovery': True, 'cv_seeds': [42]},
                 {'pair_recovery': True, 'reuse_equivalent_folds': True})
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                run_numerical_confirmation(**{**options, **values})

    def test_invalid_combined_flags_reject_without_loading_data(self):
        import tune_mnist_multiclass as example
        cases = (['--pair-recovery'], ['--pair-recovery-max-bytes', '1'],
                 ['--pair-recovery', '--no-cache', '--pair-recovery-max-bytes', '-1'],
                 ['--pair-recovery', '--no-cache', '--reuse-equivalent-folds'],
                 ['--pair-recovery', '--no-cache', '--cv-seeds', '42', '43'])
        with patch.object(example, 'load_local_multiclass_data', side_effect=AssertionError('data loaded')):
            for arguments in cases:
                with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    example.main(arguments)
                self.assertEqual(caught.exception.code, 2)

    def test_valid_preview_is_optional_and_zero_quota_is_allowed(self):
        import tune_mnist_multiclass as example
        output = io.StringIO()
        with patch.object(example, 'load_local_multiclass_data', side_effect=AssertionError('data loaded')), \
                redirect_stdout(output):
            example.main(['--pair-recovery', '--no-cache', '--pair-recovery-max-bytes', '0'])
        self.assertIn('Optional completed-pair recovery', output.getvalue())
        self.assertIn('Prepared only', output.getvalue())


if __name__ == '__main__':
    unittest.main()
