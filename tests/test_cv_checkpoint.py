"""Small fixed-fold equivalence and interruption/identity controls."""
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from runtime_test_support import synthetic_runtime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
from search_support import CandidateEvaluator, eligible_accuracy, prefix_grid
from ensemble_dwd import IneligibleSelectionError


class TinyClassifier(ClassifierMixin, BaseEstimator):
    events = []
    fail_at = None
    interrupt_at = None

    def __init__(self, offset=0.0, mode='good', random_state=42):
        self.offset = offset
        self.mode = mode
        self.random_state = random_state

    def fit(self, X, y):
        type(self).events.append(('fit', np.asarray(X).copy(), len(y)))
        count = sum(item[0] == 'fit' for item in type(self).events)
        if type(self).interrupt_at == count:
            raise KeyboardInterrupt('deliberate tiny interruption')
        if type(self).fail_at == count or self.mode == 'failure':
            raise FloatingPointError('deliberate tiny failure')
        if self.mode == 'ineligible':
            raise IneligibleSelectionError('deliberate ineligible selection')
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        self.converged_ = False
        self.criterion_reached_ = True
        self.stopping_reason_ = 'objective'
        self.n_iter_ = 3
        return self

    def predict(self, X):
        type(self).events.append(('predict', np.asarray(X).copy(), len(X)))
        if self.mode == 'prediction_failure':
            raise ValueError('deliberate prediction failure')
        return (np.asarray(X)[:, 0] > self.offset).astype(int)


def lease_process(path, connection):
    with checkpoint._Lease(path):
        connection.send('locked')
        connection.recv()
        # Bypass Python cleanup to test the OS releasing a dead owner's lock.
        os._exit(0)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / 'run'
        self.run.mkdir()
        (self.run / 'failures').mkdir()
        self.X = np.array([[-4., 0.], [-3., 1.], [-2., 3.], [-1., 5.],
                           [1., 8.], [2., 13.], [3., 21.], [4., 34.]])
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y))
        self.pipeline = Pipeline([('scale', StandardScaler()), ('model', TinyClassifier())])
        self.grid = {'model__offset': [0., .5]}
        self.runtime = synthetic_runtime(source='fixed-test-runtime')
        self.runtime_patch = patch.object(checkpoint, 'runtime_fingerprint', side_effect=lambda **kwargs: deepcopy(self.runtime))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        TinyClassifier.events = []
        TinyClassifier.fail_at = TinyClassifier.interrupt_at = None

    def session(self, *, resume=False, **kwargs):
        values = dict(run_dir=self.run, estimator=self.pipeline, X=self.X, y=self.y,
                      folds=self.folds, resume=resume, provenance={'policy': 'fixed-v1'})
        values.update(kwargs)
        return checkpoint.CheckpointedCV(**values)

    def fit_count(self):
        return sum(event[0] == 'fit' for event in TinyClassifier.events)

    def test_scores_and_fold_preprocessing_match_grid_search(self):
        expected = GridSearchCV(CandidateEvaluator(self.pipeline), prefix_grid(self.grid),
                                scoring=eligible_accuracy, cv=self.folds, refit=False,
                                n_jobs=1, error_score='raise').fit(self.X, self.y)
        TinyClassifier.events = []
        with self.session() as store:
            actual = store.search(self.grid)
        for key in ('params', 'split0_test_score', 'split1_test_score', 'mean_test_score',
                    'std_test_score', 'rank_test_score'):
            np.testing.assert_equal(actual.cv_results_[key], expected.cv_results_[key])
        fits = [event for event in TinyClassifier.events if event[0] == 'fit']
        self.assertEqual(len(fits), 4)
        for _, transformed, rows in fits:
            self.assertEqual(rows, 4)
            np.testing.assert_allclose(transformed.mean(axis=0), 0., atol=1e-15)
        self.assertEqual(actual.checkpoint_summary_['new_folds'], 4)
        self.assertEqual(actual.checkpoint_summary_['reused_folds'], 0)
        for key in ('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'):
            self.assertEqual(actual.cv_results_[key].shape, (2,))
            self.assertTrue(np.all(actual.cv_results_[key] >= 0))

    def test_same_identity_reuses_folds_and_original_diagnostics(self):
        with self.session() as store:
            first = store.search(self.grid)
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            second = store.search(self.grid)
        self.assertEqual(self.fit_count(), 0)
        self.assertEqual(second.checkpoint_summary_['reused_folds'], 4)
        self.assertEqual(first.checkpoint_summary_['diagnostics'], second.checkpoint_summary_['diagnostics'])
        facts = second.checkpoint_summary_['diagnostics'][0][0]
        self.assertIs(facts['converged_'], False)
        self.assertIs(facts['criterion_reached_'], True)
        self.assertEqual(facts['n_iter_'], 3)

    def test_partial_failure_preserves_successful_folds_and_retries_only_missing(self):
        TinyClassifier.fail_at = 3
        with self.assertRaises(FloatingPointError), self.session() as store:
            store.search(self.grid)
        self.assertEqual(len(list((self.run / 'cv-checkpoints/folds').glob('*.json'))), 2)
        TinyClassifier.fail_at = None
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            result = store.search(self.grid)
        self.assertEqual(self.fit_count(), 2)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 2)
        attempts = [json.loads(p.read_text())['payload']['state'] for p in (self.run / 'cv-checkpoints/attempts').glob('*.json')]
        self.assertIn('failed', attempts)

    def test_keyboard_interrupt_is_recorded_without_claiming_completion(self):
        TinyClassifier.interrupt_at = 2
        with self.assertRaises(KeyboardInterrupt), self.session() as store:
            store.search(self.grid)
        states = [json.loads(p.read_text())['payload']['state'] for p in (self.run / 'cv-checkpoints/attempts').glob('*.json')]
        self.assertIn('interrupted', states)
        self.assertEqual(len(list((self.run / 'cv-checkpoints/folds').glob('*.json'))), 1)
        TinyClassifier.interrupt_at = None
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            result = store.search(self.grid)
        self.assertEqual(self.fit_count(), 3)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 1)

    def test_ineligible_is_the_only_reusable_missing_score(self):
        grid = {'model__mode': ['ineligible', 'good']}
        with self.session() as store:
            result = store.search(grid)
        self.assertTrue(np.isnan(result.cv_results_['mean_test_score'][0]))
        self.assertTrue(np.isfinite(result.cv_results_['mean_test_score'][1]))
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            second = store.search(grid)
        self.assertEqual(self.fit_count(), 0)
        self.assertEqual(second.checkpoint_summary_['reused_folds'], 4)

    def test_raise_policy_does_not_turn_ineligible_into_result(self):
        with self.assertRaises(IneligibleSelectionError), self.session(error_policy='raise') as store:
            store.search({'model__mode': ['ineligible']})
        self.assertFalse(list((self.run / 'cv-checkpoints/folds').glob('*.json')))

    def test_data_fold_parameters_runtime_and_protocol_changes_reject_resume(self):
        with self.session():
            pass
        changed_folds = [(validation, train) for train, validation in self.folds]
        alternatives = ({'X': self.X + .01}, {'folds': changed_folds},
                        {'estimator': self.pipeline.set_params(model__offset=9.)},
                        {'provenance': {'policy': 'changed'}}, {'error_policy': 'raise'})
        # Each candidate uses the original pipeline unless explicitly replaced.
        self.pipeline = Pipeline([('scale', StandardScaler()), ('model', TinyClassifier())])
        for change in alternatives:
            with self.subTest(change=list(change)), self.assertRaises(checkpoint.CheckpointError):
                with self.session(resume=True, **change):
                    pass
        self.runtime['files_sha256'] = 'b' * 64
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True):
            pass

    def test_new_candidate_does_not_reuse_other_parameter_receipts(self):
        with self.session() as store:
            store.search({'model__offset': [0.]})
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            result = store.search({'model__offset': [0., .5]})
        self.assertEqual(self.fit_count(), 2)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 2)

    def test_corrupt_and_parameter_mismatched_receipts_are_not_silently_reused(self):
        with self.session() as store:
            store.search(self.grid)
        path = next((self.run / 'cv-checkpoints/folds').glob('*.json'))
        original = path.read_text()
        value = json.loads(original)
        value['payload']['score'] = .123
        path.write_text(json.dumps(value))
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.search(self.grid)
        value = json.loads(original)
        value['payload']['parameters'] = checkpoint._stable({'model__offset': 99.})
        value['sha256'] = checkpoint._digest(value['payload'])
        path.write_text(json.dumps(value))
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.search(self.grid)

    def test_failed_or_running_attempt_without_result_is_retried(self):
        TinyClassifier.fail_at = 1
        with self.assertRaises(FloatingPointError), self.session() as store:
            store.search({'model__offset': [0.]})
        path = next((self.run / 'cv-checkpoints/attempts').glob('*.json'))
        payload = json.loads(path.read_text())['payload']
        payload['state'] = 'running'  # Simulate a hard crash before terminal receipt.
        checkpoint._write_receipt(path, payload)
        TinyClassifier.fail_at = None
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            result = store.search({'model__offset': [0.]})
        self.assertEqual(self.fit_count(), 2)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 0)

    def test_exited_resume_context_can_reenter_and_replay_completed_folds(self):
        with self.session():
            pass
        store = self.session(resume=True)
        owner = store.folder / 'owner.json'
        with store:
            first_token = store.token
            first = store.search(self.grid)
        self.assertEqual(json.loads(owner.read_bytes())['state'], 'inactive')
        self.assertEqual(first.checkpoint_summary_['new_folds'], 4)
        with store:
            second_token = store.token
            self.assertNotEqual(first_token, second_token)
            active = json.loads(owner.read_bytes())
            self.assertEqual((active['state'], active['token']), ('active', second_token))
            replay = store.search(self.grid)
        self.assertEqual(self.fit_count(), 4)
        self.assertEqual(replay.checkpoint_summary_['new_folds'], 0)
        self.assertEqual(replay.checkpoint_summary_['reused_folds'], 4)
        np.testing.assert_array_equal(first.cv_results_['mean_test_score'], replay.cv_results_['mean_test_score'])
        inactive = json.loads(owner.read_bytes())
        self.assertEqual((inactive['state'], inactive['token']), ('inactive', second_token))

    def test_busy_entry_of_previously_used_context_preserves_live_owner(self):
        with self.session():
            pass
        previous = self.session(resume=True)
        with previous:
            previous.search({'model__offset': [0.]})
        with self.session(resume=True) as active:
            owner = active.folder / 'owner.json'
            before = owner.read_bytes()
            with self.assertRaises(checkpoint.CheckpointBusyError), previous:
                pass
            self.assertEqual(owner.read_bytes(), before)
            result = active.search(self.grid)
            self.assertEqual(result.checkpoint_summary_['new_folds'], 2)
            self.assertEqual(result.checkpoint_summary_['reused_folds'], 2)
            self.assertEqual(self.fit_count(), 4)

    def test_same_instance_nested_entry_leaves_outer_session_locked_and_usable(self):
        with self.session():
            pass
        with self.session(resume=True) as outer:
            owner = outer.folder / 'owner.json'
            before = owner.read_bytes()
            with self.assertRaises(checkpoint.CheckpointError), outer:
                pass
            self.assertEqual(owner.read_bytes(), before)
            with self.assertRaises(checkpoint.CheckpointBusyError), checkpoint._Lease(outer.folder / 'session.lock'):
                pass
            result = outer.search(self.grid)
            self.assertEqual(result.checkpoint_summary_['new_folds'], 4)
            self.assertEqual(self.fit_count(), 4)

    def test_repeated_exit_of_closed_context_preserves_another_active_owner(self):
        with self.session() as closed:
            pass
        with self.session(resume=True) as active:
            owner = active.folder / 'owner.json'
            before = owner.read_bytes()
            closed.__exit__(None, None, None)
            self.assertEqual(owner.read_bytes(), before)
            with self.assertRaises(checkpoint.CheckpointBusyError), checkpoint._Lease(active.folder / 'session.lock'):
                pass
            result = active.search(self.grid)
            self.assertEqual(result.checkpoint_summary_['new_folds'], 4)

    def test_failed_session_receipt_before_owner_publication_preserves_previous_owner(self):
        with self.session() as initial:
            initial.search(self.grid)
        owner = initial.folder / 'owner.json'
        sessions = initial.folder / 'sessions'
        before = owner.read_bytes()
        receipts = {path.name: path.read_bytes() for path in sessions.glob('*.json')}
        original = checkpoint._write_receipt
        failure = OSError('session receipt write failed')

        def fail_session(path, payload):
            if path.parent == sessions:
                raise failure
            return original(path, payload)

        resumed = self.session(resume=True)
        with patch.object(checkpoint, '_write_receipt', side_effect=fail_session):
            with self.assertRaises(OSError) as caught, resumed:
                pass
        self.assertIs(caught.exception, failure)
        self.assertEqual(owner.read_bytes(), before)
        self.assertEqual({path.name: path.read_bytes() for path in sessions.glob('*.json')}, receipts)
        with resumed:
            replay = resumed.search(self.grid)
        self.assertEqual(replay.checkpoint_summary_['reused_folds'], 4)
        self.assertEqual(self.fit_count(), 4)

    def test_live_worker_after_owner_publication_deactivates_attempted_session(self):
        with self.session() as initial:
            previous_token = initial.token
        owner = initial.folder / 'owner.json'
        resumed = self.session(resume=True)
        with checkpoint._Lease(initial.folder / 'locks/live-worker.lock'):
            with patch.object(checkpoint, 'write_json_atomic', wraps=checkpoint.write_json_atomic) as writes:
                with self.assertRaises(checkpoint.CheckpointBusyError), resumed:
                    pass
            published = [call.args[1] for call in writes.call_args_list
                         if call.args[0] == owner and call.args[1].get('state') == 'active']
            self.assertEqual(len(published), 1)
            attempted = published[0]['token']
            self.assertNotEqual(attempted, previous_token)
            inactive = json.loads(owner.read_bytes())
            self.assertEqual((inactive['state'], inactive['token']), ('inactive', attempted))
            session = json.loads((initial.folder / 'sessions' / (attempted + '.json')).read_bytes())
            self.assertEqual(session['payload']['owner_token'], attempted)
        with resumed:
            result = resumed.search(self.grid)
        self.assertEqual(result.checkpoint_summary_['new_folds'], 4)

    def test_interrupted_owner_write_retires_only_a_published_token(self):
        with self.session():
            pass
        resumed = self.session(resume=True)
        owner = resumed.folder / 'owner.json'
        write = checkpoint.write_json_atomic
        for published in (False, True):
            with self.subTest(published=published):
                before = owner.read_bytes()
                attempted = []
                interrupt = KeyboardInterrupt('interrupted owner publication')

                def interrupt_write(path, payload, **kwargs):
                    if path == owner and payload.get('state') == 'active':
                        attempted.append(payload['token'])
                        if published:
                            write(path, payload, **kwargs)
                        raise interrupt
                    return write(path, payload, **kwargs)

                with patch.object(checkpoint, 'write_json_atomic', side_effect=interrupt_write):
                    with self.assertRaises(KeyboardInterrupt) as caught, resumed:
                        pass
                self.assertIs(caught.exception, interrupt)
                self.assertEqual(len(attempted), 1)
                if published:
                    saved = json.loads(owner.read_bytes())
                    self.assertEqual((saved['state'], saved['token']), ('inactive', attempted[0]))
                else:
                    self.assertEqual(owner.read_bytes(), before)
                with resumed:
                    self.assertEqual(json.loads(owner.read_bytes())['token'], resumed.token)

    def test_failed_owner_read_preserves_original_publication_error(self):
        with self.session() as initial:
            pass
        resumed = self.session(resume=True)
        owner = resumed.folder / 'owner.json'
        before = owner.read_bytes()
        failure = KeyboardInterrupt('owner publication interrupted')
        write, read = checkpoint.write_json_atomic, checkpoint._read_json

        def fail_write(path, payload, **kwargs):
            if path == owner:
                raise failure
            return write(path, payload, **kwargs)

        def fail_read(path):
            if path == owner:
                raise MemoryError('owner read unavailable')
            return read(path)

        with patch.object(checkpoint, 'write_json_atomic', side_effect=fail_write), \
                patch.object(checkpoint, '_read_json', side_effect=fail_read):
            with self.assertRaises(KeyboardInterrupt) as caught, resumed:
                pass
        self.assertIs(caught.exception, failure)
        self.assertEqual(owner.read_bytes(), before)
        with resumed:
            self.assertEqual(json.loads(owner.read_bytes())['token'], resumed.token)

    def test_active_coordinator_or_worker_blocks_resume(self):
        with self.session():
            with self.assertRaises(checkpoint.CheckpointBusyError), self.session(resume=True):
                pass
        with checkpoint._Lease(self.run / 'cv-checkpoints/locks/old-worker.lock'):
            with self.assertRaises(checkpoint.CheckpointBusyError), self.session(resume=True):
                pass
        with self.session(resume=True):
            pass

    def test_dead_process_owner_lock_is_recoverable(self):
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        path = self.run / 'dead-owner.lock'
        process = context.Process(target=lease_process, args=(path, child))
        process.start()
        try:
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), 'locked')
            with self.assertRaises(checkpoint.CheckpointBusyError), checkpoint._Lease(path):
                pass
            parent.send('exit')
            process.join(30)
            self.assertEqual(process.exitcode, 0)
            with checkpoint._Lease(path):
                pass
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()
            child.close()

    def test_stale_dispatch_token_cannot_start_fitting(self):
        with self.session() as store:
            args = (store.folder, store.identity_sha256, store.runtime, 'stale-owner', 'a' * 64,
                    self.pipeline, {'model__offset': 0.}, self.X, self.y, *self.folds[0], 0, 'nan', str(self.run / 'failures'))
            with self.assertRaises(checkpoint.CheckpointBusyError):
                checkpoint._fit_fold(*args)
        self.assertEqual(self.fit_count(), 0)

    def test_worker_commits_before_parent_collects_results(self):
        captured = {}
        class LostParent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
            def __call__(self, work):
                function, args, kwargs = next(iter(work))
                if function not in (checkpoint._fit_fold, checkpoint._fit_fold_group):
                    raise AssertionError('Expected actual fold work before simulated parent loss')
                function(*args, **kwargs)
                raise KeyboardInterrupt('parent disappears after worker commit')
        # Metadata-only managed control; LostParent still receives only real folds.
        prepared = []
        class PreparationPool:
            def __init__(inner, **options):
                self.assertEqual(options, dict(n_jobs=2, pre_dispatch=2, batch_size=1, return_as='generator'))
                inner.active = False
            def __enter__(inner):
                self.assertFalse(inner.active)
                inner.active = True
                return inner
            def __exit__(inner, *unused):
                self.assertTrue(inner.active)
                inner.active = False
                prepared.append('closed')
            def __call__(inner, pending):
                self.assertTrue(inner.active)
                tasks = list(pending)
                self.assertEqual(len(tasks), 1)
                function, args, kwargs = tasks[0]
                self.assertEqual(kwargs, {})
                if function is checkpoint._execution_import_probe:
                    self.assertEqual(args, ())
                    prepared.append('import')
                    return (function(*args, **kwargs) for unused in range(1))
                self.assertIs(function, checkpoint._execution_runtime_probe)
                self.assertEqual(len(args), 1)
                prepared.append('authority')
                return (checkpoint._stable(self.runtime) for unused in range(1))
        def dispatch(**options):
            return (PreparationPool(**options) if options.get('return_as') == 'generator'
                    else LostParent(**options))
        with patch.object(checkpoint, 'Parallel', side_effect=dispatch):
            with self.assertRaises(KeyboardInterrupt), self.session(jobs=2) as store:
                store.search(self.grid)
        self.assertEqual(captured, {'n_jobs': 2, 'pre_dispatch': 2, 'batch_size': 1})
        self.assertEqual(prepared, ['import', 'authority', 'closed'])
        self.assertEqual(self.fit_count(), 1)
        self.assertEqual(len(list((self.run / 'cv-checkpoints' / 'folds').glob('*.json'))), 1)
        TinyClassifier.events = []
        with self.session(resume=True) as store:
            result = store.search(self.grid)
        self.assertEqual(result.checkpoint_summary_['reused_folds'], 1)
        self.assertEqual(self.fit_count(), 3)

    def test_completed_transaction_artifact_is_identity_bound_and_checksummed(self):
        with self.session() as store:
            self.assertIsNone(store.read_artifact('completed'))
            path = store.write_artifact('completed', {'held_out_accuracy': .75, 'final_refit_performed': True})
        with self.session(resume=True) as store:
            self.assertEqual(store.read_artifact('completed')['held_out_accuracy'], .75)
        value = json.loads(path.read_text())
        value['payload']['value']['held_out_accuracy'] = 1.
        path.write_text(json.dumps(value))
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True) as store:
            store.read_artifact('completed')

    def test_explicit_unseeded_estimator_is_rejected(self):
        self.pipeline.set_params(model__random_state=None)
        with self.assertRaisesRegex(checkpoint.CheckpointError, 'random_state'), self.session():
            pass
        self.assertEqual(self.fit_count(), 0)

    def test_random_state_state_is_preserved_in_identity(self):
        self.pipeline.set_params(model__random_state=np.random.RandomState(7))
        with self.session():
            pass
        with self.session(resume=True):
            pass
        self.pipeline.named_steps['model'].random_state.rand()
        with self.assertRaises(checkpoint.CheckpointError), self.session(resume=True):
            pass

    def test_fold_indices_and_artifact_paths_are_validated(self):
        with self.assertRaises(ValueError):
            self.session(folds=[([0, 1], [1, 2]), ([2, 3], [4, 5])])
        with self.session() as store:
            with self.assertRaises(ValueError):
                store.write_artifact('../outside', {})

    def test_caller_mutation_cannot_change_frozen_fit_data_or_estimator(self):
        with self.session() as store:
            first = store.search(self.grid)
            self.X[:] = 100.
            self.y[:] = 0
            self.pipeline.set_params(model__offset=100.)
            second = store.search(self.grid)
            np.testing.assert_equal(first.cv_results_['mean_test_score'], second.cv_results_['mean_test_score'])
            self.assertEqual(second.checkpoint_summary_['reused_folds'], 4)
            self.assertFalse(store.X.flags.writeable)
            self.assertFalse(store.y.flags.writeable)

    def test_interrupt_during_lock_acquisition_closes_owned_descriptor(self):
        path = self.run / 'interrupted-lock.lock'
        if os.name == 'nt':
            import msvcrt
            module, name, real_lock = msvcrt, 'locking', msvcrt.locking
        else:
            import fcntl
            module, name, real_lock = fcntl, 'flock', fcntl.flock
        def lock_then_interrupt(*args):
            real_lock(*args)
            raise KeyboardInterrupt('interrupted immediately after acquisition')
        with patch.object(module, name, side_effect=lock_then_interrupt):
            with self.assertRaises(KeyboardInterrupt), checkpoint._Lease(path):
                pass
        with checkpoint._Lease(path):
            pass

    def test_optional_bridge_inventory_does_not_activate_an_absent_backend(self):
        self.assertNotIn(('aocl_runtime', 'aocl-runtime'), checkpoint._runtime_package_names({}))
        self.assertIn(('aocl_runtime', 'aocl-runtime'),
                      checkpoint._runtime_package_names({'aocl_runtime': object()}))

    def test_long_run_paths_support_atomic_folds_attempts_and_resume(self):
        deep = Path(self.temporary.name)
        while len(str(deep)) < 310:
            deep /= 'nested-checkpoint-directory'
        self.assertGreater(len(str(deep)), 300)
        extended = checkpoint._io_path(deep)
        extended.mkdir(parents=True)
        (extended / 'failures').mkdir()
        # On Windows, cleanup must retain the same extended-length path form.
        if os.name == 'nt':
            import shutil
            self.addCleanup(shutil.rmtree, checkpoint._io_path(self.temporary.name))
        with self.session(run_dir=deep) as store:
            store.search(self.grid)
            store.write_artifact('completed', {'score': .5})
            attempts = list((store.folder / 'attempts').glob('*.json'))
            self.assertEqual(len(attempts), 4)
            self.assertTrue(all(len(path.stem) == 32 for path in attempts))
        TinyClassifier.events = []
        with self.session(run_dir=deep, resume=True) as store:
            resumed = store.search(self.grid)
            self.assertEqual(store.read_artifact('completed'), {'score': .5})
        self.assertEqual(resumed.checkpoint_summary_['reused_folds'], 4)
        self.assertEqual(self.fit_count(), 0)

    def test_runtime_fast_path_keeps_identity_without_binary_or_metadata_rereads(self):
        self.runtime_patch.stop()
        folder = Path(self.temporary.name) / 'fake_runtime'
        folder.mkdir()
        source = folder / '__init__.py'
        source.write_text('__version__ = "1"\n')
        binary = folder / 'backend.dll'
        binary.write_bytes(b'identity-test-only-not-executable')
        metadata = Path(self.temporary.name) / 'fake_runtime-1.dist-info' / 'METADATA'
        metadata.parent.mkdir()
        metadata.write_text('Name: fake-runtime\nVersion: 1\n')
        module = SimpleNamespace(__file__=str(source), __path__=[str(folder)], __version__='1')
        distribution = SimpleNamespace(version='1', files=['fake_runtime-1.dist-info/METADATA'],
                                       locate_file=lambda unused: metadata)
        pool = {'filepath': str(binary), 'user_api': 'blas', 'internal_api': 'fake', 'num_threads': 1, 'version': '1'}
        controller = SimpleNamespace(info=lambda: [dict(pool)])
        originals = {'_RUNTIME_FILES': None, '_RUNTIME_PACKAGE_FILES': None,
                     '_RUNTIME_PACKAGES': None, '_RUNTIME_MODULES': None,
                     '_RUNTIME_DIGEST': None, '_RUNTIME_NATIVE': None,
                     '_RUNTIME_ACCELERATOR': None, '_RUNTIME_GUARDS': set(),
                     '_RUNTIME_CONTROLLER': None, '_RUNTIME_MODULE_COUNT': None,
                     '_RESOLVED_PATHS': {}, '_HASHES': {},
                     '_RUNTIME_ARTIFACT_ALIASES': {}, '_RUNTIME_CANONICAL_ARTIFACTS': {},
                     '_RUNTIME_FILE_COUNT': None, '_RUNTIME_COMMON_FILES': None,
                     '_RUNTIME_COMMON_DIGEST': None, '_RUNTIME_COMMON_FILE_COUNT': None}
        import ensemble_dwd._cache_identity as cache_identity
        real_import = checkpoint.importlib.import_module
        def import_module(name, *args):
            return module if name == 'fake_runtime' else real_import(name, *args)
        real_open = Path.open
        with patch.multiple(checkpoint, **originals), \
                patch.object(checkpoint, '_runtime_package_names', return_value=[('fake_runtime', 'fake-runtime')]), \
                patch.dict(sys.modules, {'fake_runtime': module}), \
                patch.object(checkpoint.importlib, 'import_module', side_effect=import_module), \
                patch.object(checkpoint.importlib.metadata, 'distribution', return_value=distribution) as distributions, \
                patch.object(cache_identity, '_package_identity', return_value={'test_native': 'fixed'}) as native, \
                patch.object(checkpoint, 'ThreadpoolController', return_value=controller) as controllers, \
                patch.object(checkpoint.np, 'dot', return_value=None), \
                patch.object(Path, 'open', autospec=True, side_effect=real_open) as opens:
            first = checkpoint.runtime_fingerprint()
            reads = opens.call_count
            self.assertEqual(distributions.call_count, 1)
            second = checkpoint.runtime_fingerprint(check_inventory=False)
            third = checkpoint.runtime_fingerprint(check_inventory=False)
            self.assertEqual(first, second)
            self.assertEqual(second, third)
            self.assertEqual(opens.call_count, reads)
            self.assertEqual(distributions.call_count, 1)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(controllers.call_count, 3)
            metadata.write_text('Name: fake-runtime\nVersion: changed\n')
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.runtime_fingerprint(check_inventory=False)

    def test_cached_file_guard_detects_source_and_native_changes_without_rehash(self):
        for suffix in ('.py', '.pyd', '.dll', '.json'):
            path = Path(self.temporary.name) / ('artifact' + suffix)
            path.write_bytes(b'original')
            before = checkpoint._file_hash(path)
            with patch.object(Path, 'open', side_effect=AssertionError('Should not reread unchanged files')):
                self.assertEqual(checkpoint._file_hash(path), before)
            path.write_bytes(b'changed-and-longer')
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint._file_hash(path)


if __name__ == '__main__':
    unittest.main()
