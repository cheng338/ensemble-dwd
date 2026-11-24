"""Isolated example final-fit budgets preserve CV, receipts and restoration."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from joblib import effective_n_jobs, parallel_config
from runtime_test_support import synthetic_runtime
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import adaptive_search
import cv_checkpoint
import phase_threads
import tune_mnist
import tune_mnist_multiclass
import tune_mnist_svm
from search_support import new_run_directory


def observed_counts():
    pools = threadpool_info()
    assert any(pool['user_api'] == 'blas' for pool in pools)
    return tuple((pool['filepath'], pool['num_threads']) for pool in pools)


def observed_runtime(*, native_threads=1, **kwargs):
    # Actual pool/count observations, with explicitly synthetic source authority.
    pools = threadpool_info()
    if any(pool['num_threads'] != native_threads for pool in pools):
        raise cv_checkpoint.CheckpointError('Synthetic observer saw the wrong active budget')
    record = synthetic_runtime(source='phase-test-runtime', pools=pools)
    # Match the production full observer's stable physical-path ordering.
    # synthetic_runtime copied each row; preserve fields and duplicate rows.
    return {**record, 'native_pools': sorted(record['native_pools'],
                                           key=lambda row: row['filepath'])}


def synthetic_native_pools(files, *, native_threads):
    """Native rows for fixtures that explicitly replace the full observer.

    Install alongside that fake only. Resolve it at call time so later source
    drift/error replacements remain visible; real-observer tests use the actual
    native helper and never install this synthetic adapter.
    """
    return cv_checkpoint.runtime_fingerprint(check_inventory=False,
                                             native_threads=native_threads)['native_pools']


class PhaseTransform(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        type(self).fits.append((len(X), observed_counts()))
        return self

    def transform(self, X):
        return X


class PhaseSurface(ClassifierMixin, BaseEstimator):
    fits = []
    predictions = []
    fail_final = None

    def __init__(self, lambd=4., gamma=4., n_jobs=1):
        self.lambd, self.gamma, self.n_jobs = lambd, gamma, n_jobs

    def fit(self, X, y):
        type(self).fits.append((len(X), observed_counts()))
        if len(X) == 20 and type(self).fail_final is not None:
            raise type(self).fail_final('controlled final-fit failure')
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        type(self).predictions.append((len(X), observed_counts()))
        y = np.asarray(X)[:, 1].astype(int)
        return y if (self.lambd, self.gamma) == (4., 4.) else 1 - y


class WorkerSurface(PhaseSurface):
    events = []

    def __init__(self, lambd=4., gamma=4., n_jobs=1, base_jobs=1):
        super().__init__(lambd=lambd, gamma=gamma, n_jobs=n_jobs)
        self.base_jobs = base_jobs

    def record(self, stage, X):
        type(self).events.append((stage, len(X), self.n_jobs, self.base_jobs,
                                  effective_n_jobs(2), observed_counts()))

    def fit(self, X, y):
        self.record('fit', X)
        return super().fit(X, y)

    def predict(self, X):
        self.record('predict', X)
        return super().predict(X)


class PhaseHelperTests(unittest.TestCase):
    def test_dynamic_clone_hook_cannot_change_a_shared_component(self):
        class SharedModel:
            def __init__(self):
                self.model = WorkerSurface(n_jobs=3, base_jobs=2)

            def __getattr__(self, name):
                if name == '__sklearn_clone__':
                    return lambda: self
                return getattr(self.model, name)

            def get_params(self, deep=True):
                return self.model.get_params(deep=deep)

            def set_params(self, **options):
                self.model.set_params(**options)
                return self

        owner = SharedModel()
        for template in (owner, Pipeline([('learner_', owner)])):
            selected = clone(template)
            self.assertIs(selected['learner_'] if isinstance(selected, Pipeline) else selected, owner)
            receipt = phase_threads.configure_final_model(selected)
            self.assertEqual((owner.n_jobs, owner.base_jobs), (3, 2))
            self.assertEqual(receipt['worker_parameters'], {})
            self.assertEqual(receipt['timing_unavailable'], receipt['prediction_timing_unavailable'])

    def test_distinct_nested_steps_have_distinct_worker_records(self):
        transform = KNeighborsTransformer(n_neighbors=1, n_jobs=3)
        classifier = PhaseSurface(n_jobs=2)
        model = Pipeline([('a_', Pipeline([('b', transform)])),
                          ('a', Pipeline([('_b', classifier)]))])
        receipt = phase_threads.configure_final_model(model)
        self.assertEqual((transform.n_jobs, classifier.n_jobs), (1, 1))
        self.assertEqual(receipt['worker_parameters'],
                         {'["a_","b","n_jobs"]': 1, 'a___b__n_jobs': 1})

    def test_step_names_are_preserved_when_setting_real_worker_parameters(self):
        for name in ('n_jobs', 'base_jobs'):
            with self.subTest(step=name):
                classifier = WorkerSurface(n_jobs=3, base_jobs=2)
                inner = Pipeline([(name, classifier)])
                model = Pipeline([('base_jobs', np.str_('passthrough')), ('n_jobs', inner)])
                receipt = phase_threads.configure_final_model(model)
                self.assertIs(model['n_jobs'][name], classifier)
                self.assertEqual(model['base_jobs'], 'passthrough')
                self.assertEqual((classifier.n_jobs, classifier.base_jobs), (1, 1))
                self.assertEqual(receipt['worker_parameters'], {
                    f'n_jobs__{name}__n_jobs': 1, f'n_jobs__{name}__base_jobs': 1})
                X = np.array([[0, 0], [1, 1]])
                np.testing.assert_array_equal(model.fit(X, X[:, 1]).predict(X), X[:, 1])

    def setUp(self):
        self.threads = 1
        self.runtime = {'source': 'unchanged', 'native_pools': [{'num_threads': 1, 'filepath': 'test'}]}

        @contextmanager
        def limits(*, limits):
            previous = self.threads
            self.threads = limits
            try:
                yield
            finally:
                self.threads = previous

        def fingerprint(*, native_threads=1, **kwargs):
            self.assertEqual(self.threads, native_threads)
            result = deepcopy(self.runtime)
            result['native_pools'][0]['num_threads'] = self.threads
            return result

        contexts = (patch.object(phase_threads, 'threadpool_limits', limits),
                    patch.object(cv_checkpoint, 'runtime_fingerprint', fingerprint))
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)

    def test_success_records_observed_scope_and_restoration(self):
        receipt = {}
        with phase_threads.final_fit_threads(4, PhaseSurface(), receipt):
            self.assertEqual(self.threads, 4)
        self.assertEqual(self.threads, 1)
        self.assertEqual(receipt['state'], 'completed')
        self.assertTrue(receipt['restored'])
        self.assertEqual(receipt['before_fit']['native_pools'][0]['num_threads'], 4)
        self.assertEqual(receipt['after_fit'], receipt['before_fit'])
        self.assertEqual(receipt['after_scope'], receipt['before_scope'])

    def test_default_adds_no_runtime_observation_or_scope_change(self):
        receipt = {}
        with patch.object(cv_checkpoint, 'runtime_fingerprint', side_effect=AssertionError('unexpected scan')), \
             patch.object(phase_threads, 'threadpool_limits', side_effect=AssertionError('unexpected scope')):
            with phase_threads.final_fit_threads(1, PhaseSurface(), receipt, frozen_runtime=('existing',)):
                self.assertEqual(self.threads, 1)
        self.assertFalse(receipt['scope_changed'])
        self.assertIsNone(receipt['observed_final_runtime'])
        self.assertEqual(receipt['frozen_cv_runtime'], ('existing',))
        self.assertEqual(receipt['state'], 'completed')

    def test_exceptions_interrupts_and_exit_restore_original_budget(self):
        for error_type in (ValueError, KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type):
                receipt = {}
                with self.assertRaises(error_type):
                    with phase_threads.final_fit_threads(2, PhaseSurface(), receipt):
                        raise error_type('original failure')
                self.assertEqual(self.threads, 1)
                self.assertTrue(receipt['restored'])
                self.assertEqual(receipt['fit_exception'], error_type.__name__)

    def test_runtime_change_is_rejected_and_not_hidden_by_normalization(self):
        receipt = {}
        with self.assertRaises(cv_checkpoint.CheckpointError):
            with phase_threads.final_fit_threads(2, PhaseSurface(), receipt):
                self.runtime['source'] = 'changed'
        self.assertEqual(self.threads, 1)
        self.assertFalse(receipt['restored'])
        self.assertIn('restoration_error', receipt)

    def test_original_interrupt_survives_observer_failure(self):
        receipt = {}
        with self.assertRaisesRegex(KeyboardInterrupt, 'original'):
            with phase_threads.final_fit_threads(2, PhaseSurface(), receipt):
                self.runtime['source'] = 'changed'
                raise KeyboardInterrupt('original')
        self.assertEqual(self.threads, 1)
        self.assertIn('observation_error', receipt)
        self.assertIn('restoration_error', receipt)

    def test_wrong_frozen_runtime_rejects_before_fit(self):
        with self.assertRaises(cv_checkpoint.CheckpointError):
            with phase_threads.final_fit_threads(2, PhaseSurface(), {}, frozen_runtime=('bad',)):
                self.fail('Fit must not run')
        self.assertEqual(self.threads, 1)

    def test_invalid_values_and_nested_parallel_settings(self):
        for value in (True, False, np.bool_(True), 0, -1, 1.5, '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                phase_threads.validate_final_native_threads(value)
        self.assertEqual(phase_threads.validate_final_native_threads(np.int64(2)), 2)
        for jobs in (None, -1, 2, True):
            pipeline = Pipeline([('nested', PhaseSurface(n_jobs=jobs))])
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                phase_threads.validate_final_native_threads(2, pipeline)
            self.assertEqual(phase_threads.validate_final_native_threads(1, pipeline), 1)


class PhaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        rows = np.arange(20)
        self.X = np.column_stack((rows, rows % 2))
        self.y = rows % 2
        test_rows = np.arange(100, 108)
        self.test_X = np.column_stack((test_rows, test_rows % 2))
        self.test_y = test_rows % 2
        self.folds = [(np.setdiff1d(rows, rows[i:i+4]), rows[i:i+4]) for i in range(0, 20, 4)]
        self.pipeline = Pipeline([('transform', PhaseTransform()), ('ensemble', PhaseSurface())])
        self.grid = {'ensemble__lambd': [1., 4., 16.], 'ensemble__gamma': [1., 4., 16.]}
        self.config = adaptive_search.AdaptiveSearchConfig(refine=False, max_candidates=9)
        contexts = (patch.object(cv_checkpoint, 'runtime_fingerprint', side_effect=observed_runtime),
                    patch.object(cv_checkpoint, '_native_pool_fingerprint', synthetic_native_pools))
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)
        self.reset_events()

    def reset_events(self):
        PhaseTransform.fits = []
        PhaseSurface.fits = []
        PhaseSurface.predictions = []
        PhaseSurface.fail_final = None

    def run_workflow(self, name='run', *, legacy=False, resume=False, **kwargs):
        run_dir = new_run_directory(self.base / name, resume=resume)
        with redirect_stdout(io.StringIO()):
            if legacy:
                return tune_mnist.run_search_workflow(
                    self.pipeline, self.grid, self.X, self.y, self.test_X, self.test_y,
                    self.folds, run_dir, verbose=0, **kwargs)
            return adaptive_search.run_adaptive_search(
                self.pipeline, self.grid, self.X, self.y, self.test_X, self.test_y,
                self.folds, run_dir, config=self.config, resume=resume, **kwargs)

    def assert_scope_events(self):
        for events in (PhaseTransform.fits, PhaseSurface.fits):
            self.assertTrue(any(n == 20 for n, pools in events))
            for n, pools in events:
                self.assertTrue(all(value == 1 for path, value in pools))
        self.assertTrue(any(n == 8 for n, pools in PhaseSurface.predictions))
        self.assertTrue(all(value == 1 for n, pools in PhaseSurface.predictions for path, value in pools))

    def test_adaptive_scopes_full_pipeline_and_records_metadata(self):
        with threadpool_limits(limits=3):
            before = observed_counts()
            result = self.run_workflow(final_native_threads=1)
            self.assertEqual(observed_counts(), before)
        self.assert_scope_events()
        settings = json.loads((self.base / 'run/run.json').read_text())['settings']
        self.assertEqual(settings['native_threads'], 1)
        self.assertEqual(settings['final_native_threads'], 1)
        self.assertEqual(settings['held_out_native_threads'], 1)
        receipt = result['refit']['native_runtime']
        self.assertEqual(receipt['requested_native_threads'], 1)
        self.assertEqual(receipt['estimator_workers'], 1)
        self.assertEqual(receipt['worker_parameters'], {'ensemble__n_jobs': 1})

    def test_legacy_scopes_full_pipeline_and_restores_caller(self):
        with threadpool_limits(limits=3):
            before = observed_counts()
            self.run_workflow(legacy=True, final_native_threads=1)
            self.assertEqual(observed_counts(), before)
        self.assert_scope_events()
        refit = json.loads((self.base / 'run/refit.json').read_text())
        self.assertEqual(refit['native_runtime']['requested_native_threads'], 1)

    def test_implicit_and_explicit_default_match_fit_and_prediction_events(self):
        first = self.run_workflow('implicit')
        events = deepcopy((PhaseTransform.fits, PhaseSurface.fits, PhaseSurface.predictions))
        self.reset_events()
        second = self.run_workflow('explicit', final_native_threads=1)
        self.assertEqual(events, (PhaseTransform.fits, PhaseSurface.fits, PhaseSurface.predictions))
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['completed']['held_out_accuracy'], second['completed']['held_out_accuracy'])
        self.assert_scope_events()

    def test_final_failures_interrupts_and_exit_restore_both_scopes(self):
        for legacy in (False, True):
            for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                with self.subTest(legacy=legacy, error_type=error_type):
                    self.reset_events()
                    PhaseSurface.fail_final = error_type
                    name = f'failure-{legacy}-{error_type.__name__}'
                    with threadpool_limits(limits=3):
                        before = observed_counts()
                        with self.assertRaises(error_type):
                            self.run_workflow(name, legacy=legacy, final_native_threads=1)
                        self.assertEqual(observed_counts(), before)
                    self.assertFalse(any(n == 8 for n, pools in PhaseSurface.predictions))
                    status = json.loads((self.base / name / 'status.json').read_text())
                    self.assertEqual(status['phase'], 'final_refit')
                    if error_type is not ValueError or legacy:
                        self.assertEqual(status['final_runtime']['state'], 'failed')
                    else:
                        refit = json.loads((self.base / name / 'refit.json').read_text())
                        self.assertEqual(refit['native_runtime']['state'], 'failed')

    def test_interrupted_final_resume_retains_cv_and_completed_replay_is_idle(self):
        PhaseSurface.fail_final = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_workflow(final_native_threads=1)
        cv_fits = [event for event in PhaseSurface.fits if event[0] != 20]
        PhaseSurface.fail_final = None
        result = self.run_workflow(resume=True, final_native_threads=1)
        self.assertEqual(cv_fits, [event for event in PhaseSurface.fits if event[0] != 20])
        counts = (len(PhaseSurface.fits), len(PhaseSurface.predictions))
        replay = self.run_workflow(resume=True, final_native_threads=1)
        self.assertTrue(replay['resumed_completed'])
        self.assertEqual(result['refit'], replay['refit'])
        self.assertEqual(counts, (len(PhaseSurface.fits), len(PhaseSurface.predictions)))

    def test_changed_final_budget_rejects_resume_without_status_mutation(self):
        self.run_workflow(final_native_threads=1)
        before = (self.base / 'run/status.json').read_bytes()
        counts = (len(PhaseSurface.fits), len(PhaseSurface.predictions))
        with self.assertRaisesRegex(ValueError, 'final_native_threads'):
            self.run_workflow(resume=True, final_native_threads=2)
        self.assertEqual((self.base / 'run/status.json').read_bytes(), before)
        self.assertEqual(counts, (len(PhaseSurface.fits), len(PhaseSurface.predictions)))

    def test_changed_phase_helper_hash_rejects_completed_resume(self):
        self.run_workflow(final_native_threads=1)
        before = (self.base / 'run/status.json').read_bytes()
        counts = (len(PhaseSurface.fits), len(PhaseSurface.predictions))
        with patch.object(adaptive_search, 'final_thread_policy_sha256', return_value='changed-helper'):
            with self.assertRaises(cv_checkpoint.CheckpointError):
                self.run_workflow(resume=True, final_native_threads=1)
        self.assertEqual((self.base / 'run/status.json').read_bytes(), before)
        self.assertEqual(counts, (len(PhaseSurface.fits), len(PhaseSurface.predictions)))

    def test_invalid_budget_rejects_before_receipts_or_fits(self):
        for legacy in (False, True):
            name = 'invalid-' + str(legacy)
            with self.assertRaises(ValueError):
                self.run_workflow(name, legacy=legacy, final_native_threads=True)
            self.assertFalse((self.base / name / 'run.json').exists())
            self.assertEqual(PhaseSurface.fits, [])

    def test_cv_preserves_worker_options_and_final_fit_and_prediction_use_one(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.reset_events()
                WorkerSurface.events = []
                candidate = WorkerSurface(n_jobs=3, base_jobs=4)
                self.pipeline.set_params(ensemble=candidate)
                self.grid['ensemble__n_jobs'] = [2]
                with parallel_config(backend='threading', n_jobs=3), threadpool_limits(3):
                    before = observed_counts()
                    result = self.run_workflow('workers-' + str(legacy), legacy=legacy)
                    self.assertEqual(observed_counts(), before)
                    self.assertEqual(effective_n_jobs(None), 3)
                final = [row for row in WorkerSurface.events
                         if (row[0], row[1]) in {('fit', 20), ('predict', 8)}]
                self.assertEqual([row[:5] for row in final],
                                 [('fit', 20, 1, 1, 1), ('predict', 8, 1, 1, 1)])
                self.assertTrue(all(n == 1 for row in final for _, n in row[5]))
                cv = [row for row in WorkerSurface.events if row[0] == 'fit' and row[1] < 20]
                self.assertTrue(cv)
                self.assertTrue(all(row[2:4] == (2, 4) for row in cv))
                self.assertEqual((candidate.n_jobs, candidate.base_jobs), (3, 4))
                self.assertEqual(self.grid['ensemble__n_jobs'], [2])
                self.assertEqual(result['selection']['parameters']['ensemble__n_jobs'], 2)


class PhaseCLITests(unittest.TestCase):
    def test_three_previews_accept_option_without_loading_data(self):
        for module in (tune_mnist, tune_mnist_multiclass, tune_mnist_svm):
            with self.subTest(module=module.__name__):
                output = io.StringIO()
                with redirect_stdout(output):
                    module.main(['--final-native-threads', '1'])
                self.assertIn('CV 1 per worker; final pipeline fit 1; held-out scoring 1', output.getvalue())

    def test_three_previews_reject_unsupported_budgets(self):
        for module in (tune_mnist, tune_mnist_multiclass, tune_mnist_svm):
            for value in ('0', '2'):
                with self.subTest(module=module.__name__, value=value), patch('sys.stderr', io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        module.main(['--final-native-threads', value])
                    self.assertEqual(raised.exception.code, 2)

    def test_three_run_entrypoints_forward_budget_to_shared_workflow(self):
        class StopBeforeSearch(Exception):
            pass

        with tempfile.TemporaryDirectory() as folder:
            for module in (tune_mnist, tune_mnist_multiclass, tune_mnist_svm):
                with self.subTest(module=module.__name__):
                    rows = np.arange(100)
                    X = np.column_stack((rows, rows % 10))
                    y = rows % (10 if module is tune_mnist_multiclass else 2)
                    if module is tune_mnist_multiclass:
                        loader, loaded = 'load_local_multiclass_data', (X, X.copy(), y, y.copy(), {})
                    else:
                        loader, loaded = 'load_local_data', (X, X.copy(), y, y.copy())
                    with patch.object(module, loader, return_value=loaded), \
                         patch.object(adaptive_search, 'run_adaptive_search', side_effect=StopBeforeSearch) as run, \
                         redirect_stdout(io.StringIO()):
                        with self.assertRaises(StopBeforeSearch):
                            module.main(['--data', 'fixture.npz', '--run', '--no-cache',
                                         '--output-dir', str(Path(folder) / module.__name__),
                                         '--final-native-threads', '1'])
                    self.assertEqual(run.call_args.kwargs['final_native_threads'], 1)


class RealPhaseRuntimeTests(unittest.TestCase):
    def setUp(self):
        from ensemble_dwd._cache import _initialize_numerical_runtime
        _initialize_numerical_runtime()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.X = np.column_stack((np.linspace(-2., 2., 48), np.sin(np.arange(48)), (np.arange(48) % 5) / 8.))
        self.y = np.where(self.X[:, 0] < 0., -3, 7)

    def binary(self, *, cache=True):
        from ensemble_dwd import EnsembleKernelDWD
        return EnsembleKernelDWD(M=2, theta=.5, sv_selection_ratio=1., lambd=.05,
                                 gamma=.2, max_iter=3, stopping='fixed', random_state=17,
                                 cache_dir=self.folder.name if cache else None,
                                 store_base_estimators=True)

    def test_real_cache_separates_budgets_and_reuses_restored_one_thread_entries(self):
        models, receipts = [], []
        with threadpool_limits(limits=1):
            runtime = cv_checkpoint.runtime_fingerprint()
            frozen = cv_checkpoint._stable(runtime)
            for count in (1, 2, 2, 1):
                model, receipt = self.binary(), {}
                with phase_threads.final_fit_threads(count, model, receipt, frozen_runtime=frozen):
                    model.fit(self.X, self.y)
                self.assertTrue(all(value == 1 for path, value in observed_counts()))
                self.assertEqual(cv_checkpoint.runtime_fingerprint(), runtime)
                models.append(model)
                receipts.append(receipt)
            self.assertEqual([model.cache_hit_ for model in models], [False, False, True, True])
            self.assertNotEqual(models[0].cache_key_, models[1].cache_key_)
            self.assertEqual(models[1].cache_key_, models[2].cache_key_)
            self.assertEqual(models[0].cache_key_, models[3].cache_key_)
            for first, second in ((models[0], models[3]), (models[1], models[2])):
                np.testing.assert_array_equal(first.consensus_scores_, second.consensus_scores_)
                np.testing.assert_array_equal(first.selected_indices_, second.selected_indices_)
                np.testing.assert_array_equal(first.final_estimator_.dual_coef_, second.final_estimator_.dual_coef_)
                np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
            self.assertTrue(all(receipts[i]['restored'] for i in (1, 2)))

    def test_live_helper_file_guard_detects_isolated_copy_replacement(self):
        helper = Path(phase_threads.__file__).absolute()
        copy = Path(self.folder.name) / 'phase_threads_copy.py'
        copy.write_bytes(helper.read_bytes())
        original_hash = cv_checkpoint._file_hash

        def redirected_hash(path):
            return original_hash(copy if Path(path).absolute() == helper else path)

        with threadpool_limits(limits=1):
            baseline = cv_checkpoint.runtime_fingerprint()
            with patch.object(cv_checkpoint, '_file_hash', side_effect=redirected_hash):
                self.assertEqual(cv_checkpoint.runtime_fingerprint(check_inventory=False), baseline)
                copy.write_bytes(copy.read_bytes() + b'\n# isolated guard mutation\n')
                with self.assertRaises(cv_checkpoint.CheckpointError):
                    cv_checkpoint.runtime_fingerprint(check_inventory=False)
            self.assertEqual(cv_checkpoint.runtime_fingerprint(check_inventory=False), baseline)

    def test_real_sequential_ovo_fit_observes_wider_budget_in_all_pairs(self):
        from ensemble_dwd import OvOClassifier
        import ensemble_dwd.multiclass as multiclass
        from sklearn.preprocessing import StandardScaler
        centers = np.array([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]])
        offsets = np.column_stack((np.linspace(-.2, .2, 12), np.sin(np.arange(12)) / 8., np.arange(12) / 64.))
        X = np.vstack([center + offsets for center in centers])
        y = np.repeat(['a', 'b', 'c'], 12)
        model = Pipeline([('scale', StandardScaler()), ('ovo', OvOClassifier(self.binary(cache=False), n_jobs=1))])
        original = multiclass._fit_pair
        pair_counts = []

        def observed_pair(*args, **kwargs):
            pair_counts.append(observed_counts())
            return original(*args, **kwargs)

        with threadpool_limits(limits=1):
            frozen = cv_checkpoint._stable(cv_checkpoint.runtime_fingerprint())
            receipt = {}
            with patch.object(multiclass, '_fit_pair', side_effect=observed_pair):
                with phase_threads.final_fit_threads(2, model, receipt, frozen_runtime=frozen):
                    model.fit(X, y)
            self.assertEqual(len(pair_counts), 3)
            self.assertTrue(all(value == 2 for counts in pair_counts for path, value in counts))
            self.assertTrue(all(value == 1 for path, value in observed_counts()))
            self.assertTrue(receipt['restored'])
            self.assertEqual(model.predict(X).shape, y.shape)
            self.assertEqual(len(model.named_steps['ovo'].estimators_), 3)


if __name__ == '__main__':
    unittest.main()
