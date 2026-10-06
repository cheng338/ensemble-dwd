"""Base-stage reuse, invalidation, and storage contracts."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone

from ensemble_dwd import EnsembleKernelDWD
from ensemble_dwd._cache import _cached_base_stage, _cache_path


class CacheProbeDWD:
    """A deterministic affine model whose actual fit calls can be counted."""

    fits = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.fit_number = len(type(self).fits)
        type(self).fits.append(self)
        self.X = np.array(X, copy=True)
        self.classes_ = np.unique(y)
        self.n_iter_ = 7
        self.converged_ = True
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, 0]


def constant_kernel(X, Y):
    return np.ones((len(X), len(Y)))


def cache_process_worker(cache_dir, count_path, start_path, ready_path):
    """Independent Python-process target, including on Windows."""
    try:
        Path(ready_path).write_text('ready', encoding='utf-8')
        deadline = time.monotonic() + 20
        while not Path(start_path).is_file():
            if time.monotonic() > deadline:
                raise RuntimeError('The cache worker start signal was not received.')
            time.sleep(.01)

        def compute():
            with open(count_path, 'a', encoding='utf-8') as stream:
                stream.write('computed\n')
                stream.flush()
            time.sleep(.1)
            return {'answer': 37}

        value, hit, key = _cached_base_stage(cache_dir, {'shared': 'concurrent'}, compute)
        print(json.dumps({'value': value, 'hit': hit, 'key': key}))
    except Exception as error:
        print(json.dumps({'error': type(error).__name__ + ': ' + str(error)}))
        raise SystemExit(1)


def fresh_real_cache_worker(cache_dir):
    """Exercise cache provenance before the first actual BLAS-backed fit."""
    from threadpoolctl import threadpool_info

    # Native thread counts come from the child environment. Calling threadpool
    # controllers before fit could itself initialize the runtime being tested.
    X = np.column_stack((np.arange(24, dtype=float) - 11.5, np.arange(24) % 3)) / 6.
    y = np.zeros(24, dtype=int)
    # Put one rare-class row in each fixed seed-17 disjoint block; this
    # fixture tests runtime cache provenance, not sampling eligibility.
    y[[10, 23]] = 1
    settings = dict(M=2, theta=.5, sv_selection_ratio=1., lambd=.05, gamma=.2,
                    max_iter=8, stopping='fixed', random_state=17, cache_dir=cache_dir)
    first = EnsembleKernelDWD(**settings).fit(X, y)
    pools_after_first = threadpool_info()
    second = EnsembleKernelDWD(**settings).fit(X, y)
    same_predictions = bool(np.array_equal(first.predict(X), second.predict(X)))
    same_consensus = bool(np.array_equal(first.consensus_scores_, second.consensus_scores_))
    print(json.dumps({
        'first_hit': first.cache_hit_, 'second_hit': second.cache_hit_,
        'first_key': first.cache_key_, 'second_key': second.cache_key_,
        'same_predictions': same_predictions, 'same_consensus': same_consensus,
        'first_implementation': first.final_estimator_.implementation,
        'second_implementation': second.final_estimator_.implementation,
        'pool_prefixes_after_first': [item['prefix'] for item in pools_after_first],
        'entry_count': len(list((Path(cache_dir) / 'base-stage-v2').glob('*.joblib'))),
    }))


class CacheEstimatorTests(unittest.TestCase):
    def setUp(self):
        CacheProbeDWD.fits = []
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.X = np.column_stack((np.linspace(-2., 2., 20), np.cos(np.arange(20))))
        self.y = np.tile(['left', 'right'], 10)

    def model(self, **kwargs):
        params = dict(M=3, theta=.3, sv_selection_ratio=.3, gamma=.02,
                      random_state=17, cache_dir=self.directory.name)
        params.update(kwargs)
        return EnsembleKernelDWD(**params)

    def fit_spy(self, model, X=None, y=None):
        with patch('ensemble_dwd.estimator.KernGDWD', CacheProbeDWD):
            return model.fit(self.X if X is None else X, self.y if y is None else y)

    def test_retention_change_reuses_bases_but_always_fits_a_new_final_model(self):
        first = self.fit_spy(self.model())
        self.assertFalse(first.cache_hit_)
        self.assertEqual(len(CacheProbeDWD.fits), 4)
        second = self.fit_spy(self.model(sv_selection_ratio=.5))
        self.assertTrue(second.cache_hit_)
        self.assertEqual(first.cache_key_, second.cache_key_)
        self.assertEqual(len(CacheProbeDWD.fits), 5)
        self.assertIsNot(first.final_estimator_, second.final_estimator_)
        self.assertEqual(len(first.selected_indices_), 6)
        self.assertEqual(len(second.selected_indices_), 10)
        assert_array_equal(first.consensus_scores_, second.consensus_scores_)
        for a, b in zip(first.subset_indices_, second.subset_indices_):
            assert_array_equal(a, b)
        for model in [*first.estimators_, first.final_estimator_, second.final_estimator_]:
            self.assertEqual(model.params['implementation'], 'optimized')

    def test_base_parameter_changes_invalidate_the_entry(self):
        baseline = self.fit_spy(self.model())
        changes = [dict(M=2), dict(theta=.25), dict(lambd=.02), dict(q=2.),
                   dict(gamma=.04), dict(kernel='linear'), dict(max_iter=11),
                   dict(obj_tol=2e-5), dict(stopping='fixed'), dict(tol=2e-6),
                   dict(acceleration='restart'), dict(random_state=19),
                   dict(residual_check_order='adaptive'),
                   dict(affine_computation='joint'),
                   dict(exact_recovery='extended'),
                   dict(prediction_batch_size=3), dict(store_base_estimators=False)]
        for change in changes:
            with self.subTest(change=change):
                model = self.fit_spy(self.model(**change))
                self.assertFalse(model.cache_hit_)
                self.assertNotEqual(model.cache_key_, baseline.cache_key_)

    def test_input_values_row_order_and_labels_invalidate_the_entry(self):
        baseline = self.fit_spy(self.model())
        for X, y in [(self.X + .25, self.y), (self.X[::-1], self.y[::-1]),
                     (self.X, np.tile(['negative', 'positive'], 10))]:
            with self.subTest(first_row=X[0].tolist(), first_label=y[0]):
                model = self.fit_spy(self.model(), X=X, y=y)
                self.assertFalse(model.cache_hit_)
                self.assertNotEqual(model.cache_key_, baseline.cache_key_)

    def test_ignored_gamma_does_not_split_linear_kernel_cache_entries(self):
        first = self.fit_spy(self.model(kernel='linear', gamma=.02))
        second = self.fit_spy(self.model(kernel='linear', gamma=.7))
        self.assertTrue(second.cache_hit_)
        self.assertEqual(first.cache_key_, second.cache_key_)
        self.assertEqual(len(CacheProbeDWD.fits), 5)

    def test_automatic_scale_and_matching_numeric_gamma_share_effective_key(self):
        first = self.fit_spy(self.model(gamma='scale'))
        second = self.fit_spy(self.model(gamma=first.gamma_))
        self.assertTrue(second.cache_hit_)
        self.assertEqual(first.cache_key_, second.cache_key_)

    def test_nondeterministic_or_callable_models_bypass_the_cache(self):
        for options in [dict(random_state=None, M=1, theta=1.), dict(kernel=constant_kernel)]:
            with self.subTest(options=options):
                CacheProbeDWD.fits = []
                first = self.fit_spy(self.model(**options))
                second = self.fit_spy(self.model(**options))
                self.assertFalse(first.cache_hit_)
                self.assertFalse(second.cache_hit_)
                self.assertIsNone(first.cache_key_)
                self.assertIsNone(second.cache_key_)
                self.assertEqual(len(CacheProbeDWD.fits), 2 * (second.M + 1))
                self.assertIsInstance(second.cache_status_, str)

    def test_default_cache_off_recomputes_and_is_cloneable(self):
        model = self.model(cache_dir=None)
        copied = clone(model)
        self.assertEqual(copied.get_params(), model.get_params())
        first = self.fit_spy(model)
        second = self.fit_spy(copied)
        self.assertFalse(first.cache_hit_)
        self.assertFalse(second.cache_hit_)
        self.assertEqual(len(CacheProbeDWD.fits), 8)

    def test_changed_runtime_provenance_invalidates_estimator_cache(self):
        with patch('ensemble_dwd.estimator._runtime_provenance', return_value={'runtime': 'a'}):
            first = self.fit_spy(self.model())
        with patch('ensemble_dwd.estimator._runtime_provenance', return_value={'runtime': 'b'}):
            second = self.fit_spy(self.model())
        self.assertFalse(first.cache_hit_)
        self.assertFalse(second.cache_hit_)
        self.assertNotEqual(first.cache_key_, second.cache_key_)
        self.assertEqual(len(CacheProbeDWD.fits), 8)

    def test_custom_subclasses_bypass_cache(self):
        class DerivedEnsemble(EnsembleKernelDWD):
            pass

        for _ in range(2):
            model = DerivedEnsemble(**self.model().get_params())
            self.fit_spy(model)
            self.assertFalse(model.cache_hit_)
            self.assertIsNone(model.cache_key_)
        self.assertEqual(len(CacheProbeDWD.fits), 8)

    def test_lean_cache_roundtrip_retains_diagnostics_and_prediction(self):
        first = self.fit_spy(self.model(store_base_estimators=False))
        second = self.fit_spy(self.model(store_base_estimators=False, sv_selection_ratio=.5))
        self.assertTrue(second.cache_hit_)
        for model in [first, second]:
            self.assertIsNone(model.estimators_)
            self.assertEqual(len(model.subset_indices_), model.M)
            assert_array_equal(model.n_iter_, np.full(model.M + 1, 7))
            assert_array_equal(model.converged_, np.ones(model.M + 1, dtype=bool))
            restored = pickle.loads(pickle.dumps(model))
            self.assertIsNone(restored.estimators_)
            assert_array_equal(restored.predict(self.X), model.predict(self.X))
            assert_array_equal(restored.decision_function(self.X), model.decision_function(self.X))


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = 0

    def compute(self):
        self.calls += 1
        return {'answer': 37, 'values': np.arange(4)}

    def test_long_cache_directory_supports_atomic_miss_and_hit(self):
        directory = Path(self.directory.name)
        while len(str(directory)) < 310:
            directory /= 'nested-cache-directory'
        self.assertGreater(len(str(directory)), 300)
        if os.name == 'nt':
            import shutil
            self.addCleanup(shutil.rmtree, _cache_path(self.directory.name))
        first, hit, key = _cached_base_stage(str(directory), {'long': 'path'}, self.compute)
        self.assertFalse(hit)
        second, hit, second_key = _cached_base_stage(directory, {'long': 'path'}, self.compute)
        self.assertTrue(hit)
        self.assertEqual(second_key, key)
        self.assertEqual(self.calls, 1)
        assert_array_equal(first['values'], second['values'])
        self.assertEqual(second['answer'], 37)
        stored = _cache_path(directory) / 'base-stage-v2'
        self.assertTrue((stored / (key + '.joblib')).is_file())
        self.assertTrue((stored / (key + '.json')).is_file())
        self.assertEqual(list(stored.glob('*.tmp')), [])

    def test_key_changes_when_runtime_identity_changes(self):
        _, first_hit, first_key = _cached_base_stage(
            self.directory.name, {'data': [1, 2], 'runtime': 'first'}, self.compute)
        _, second_hit, second_key = _cached_base_stage(
            self.directory.name, {'data': [1, 2], 'runtime': 'second'}, self.compute)
        self.assertFalse(first_hit)
        self.assertFalse(second_hit)
        self.assertNotEqual(first_key, second_key)
        self.assertEqual(self.calls, 2)

    def test_failed_compute_does_not_publish_a_successful_entry(self):
        def failing():
            raise FloatingPointError('injected base-stage failure')

        with self.assertRaisesRegex(FloatingPointError, 'injected'):
            _cached_base_stage(self.directory.name, {'failure': 'retry'}, failing)
        value, hit, _ = _cached_base_stage(self.directory.name, {'failure': 'retry'}, self.compute)
        self.assertFalse(hit)
        self.assertEqual(value['answer'], 37)
        self.assertEqual(self.calls, 1)

    def test_payload_checksum_mismatch_recomputes_then_hits(self):
        _, _, key = _cached_base_stage(self.directory.name, {'corruption': 'payload'}, self.compute)
        payload = Path(self.directory.name) / 'base-stage-v2' / (key + '.joblib')
        payload.write_bytes(b'interrupted payload')
        value, hit, second_key = _cached_base_stage(
            self.directory.name, {'corruption': 'payload'}, self.compute)
        self.assertFalse(hit)
        self.assertEqual(second_key, key)
        self.assertEqual(value['answer'], 37)
        self.assertEqual(self.calls, 2)
        _, hit, _ = _cached_base_stage(self.directory.name, {'corruption': 'payload'}, self.compute)
        self.assertTrue(hit)
        self.assertEqual(self.calls, 2)

    def test_invalid_metadata_contents_are_recomputed(self):
        for corrupted in ['not json', '[]', 'null', '"unexpected scalar"']:
            with self.subTest(corrupted=corrupted):
                key_data = {'corruption': corrupted}
                _, _, key = _cached_base_stage(self.directory.name, key_data, self.compute)
                metadata = Path(self.directory.name) / 'base-stage-v2' / (key + '.json')
                metadata.write_text(corrupted, encoding='utf-8')
                before = self.calls
                value, hit, _ = _cached_base_stage(self.directory.name, key_data, self.compute)
                self.assertFalse(hit)
                self.assertEqual(value['answer'], 37)
                self.assertEqual(self.calls, before + 1)
                self.assertIsInstance(json.loads(metadata.read_text()), dict)

    def test_first_real_fit_and_second_fit_share_key_in_fresh_process(self):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(sys.path)
        for name in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment[name] = '4'
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--cache-real-worker',
             self.directory.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        result = json.loads(completed.stdout)
        self.assertFalse(result['first_hit'], result)
        self.assertTrue(result['second_hit'], result)
        self.assertEqual(result['first_key'], result['second_key'], result)
        self.assertEqual(result['entry_count'], 1, result)
        self.assertTrue(result['same_predictions'], result)
        self.assertTrue(result['same_consensus'], result)
        self.assertEqual(result['first_implementation'], 'optimized')
        self.assertEqual(result['second_implementation'], 'optimized')

    def test_concurrent_processes_compute_one_entry_once(self):
        folder = Path(self.directory.name)
        count_path = str(folder / 'compute-count.txt')
        start_path = folder / 'start'
        ready_paths = [folder / f'worker-{number}.ready' for number in range(3)]
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(sys.path)
        for name in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment[name] = '1'
        processes = []
        try:
            for ready_path in ready_paths:
                processes.append(subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--cache-worker',
                     self.directory.name, count_path, str(start_path), str(ready_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env=environment, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
            deadline = time.monotonic() + 20
            while not all(path.is_file() for path in ready_paths):
                self.assertLess(time.monotonic(), deadline, 'Cache workers did not start.')
                time.sleep(.01)
            start_path.write_text('start', encoding='utf-8')
            rows = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stderr + stdout)
                rows.append(json.loads(stdout))
            self.assertFalse([row for row in rows if 'error' in row], rows)
            self.assertEqual(Path(count_path).read_text().splitlines(), ['computed'])
            self.assertEqual(sum(not row['hit'] for row in rows), 1)
            self.assertEqual(len({row['key'] for row in rows}), 1)
            self.assertTrue(all(row['value'] == {'answer': 37} for row in rows))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=10)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--cache-worker':
        cache_process_worker(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == '--cache-real-worker':
        fresh_real_cache_worker(*sys.argv[2:])
    else:
        unittest.main()
