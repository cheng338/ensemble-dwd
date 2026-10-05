"""Public search output paths and atomic-write failure behavior, without fits."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import search_support as support


class PathWorkflowClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic test fixture that does not run a numerical optimizer."""
    fits = 0
    predictions = 0

    def __init__(self, lambd=1.0, gamma=1.0, random_state=42):
        self.lambd, self.gamma, self.random_state = lambd, gamma, random_state

    def fit(self, X, y):
        type(self).fits += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        type(self).predictions += 1
        return (X[:, 0] > 0).astype(int)


def ordinary_path(path):
    value = str(path)
    if os.name == 'nt' and value.startswith('\\\\?\\UNC\\'):
        value = '\\\\' + value[8:]
    elif os.name == 'nt' and value.startswith('\\\\?\\'):
        value = value[4:]
    return Path(value)


class PublicOutputPathTests(unittest.TestCase):
    def setUp(self):
        # Give the standard-library cleanup an extended root too, so a failed
        # assertion cannot leave deep temporary test directories behind.
        self.temporary = tempfile.TemporaryDirectory(
            prefix='public-output-', dir=support._io_path(tempfile.gettempdir()))
        self.addCleanup(self.temporary.cleanup)
        # A temporary root may contain an 8.3 alias such as RUNNER~1. Run
        # directories resolve those aliases; keep the fixture's expected root
        # canonical while the write tests still check requested spelling.
        self.root = ordinary_path(Path(self.temporary.name).resolve())

    def read(self, path):
        return json.loads(support._io_path(path).read_text(encoding='utf-8'))

    def test_atomic_replacement_and_explicit_no_overwrite(self):
        path = self.root / 'status.json'
        self.assertEqual(support.write_json_atomic(path, {'state': 'running'}), path)
        with patch.object(support.tempfile, 'NamedTemporaryFile') as temporary:
            with self.assertRaises(FileExistsError):
                support.write_json_atomic(path, {'state': 'discarded'})
            temporary.assert_not_called()
        self.assertEqual(self.read(path), {'state': 'running'})
        self.assertEqual(support.write_json_atomic(path, {'state': 'completed'}, replace=True), path)
        self.assertEqual(self.read(path), {'state': 'completed'})
        self.assertEqual(list(support._io_path(self.root).glob('.record-*')), [])

    def test_failed_replace_preserves_existing_record_and_cleans_temporary(self):
        path = self.root / 'status.json'
        support.write_json_atomic(path, {'state': 'running'})
        failure = PermissionError('injected replacement failure')
        with patch.object(support.os, 'replace', side_effect=failure):
            with self.assertRaises(PermissionError) as caught:
                support.write_json_atomic(path, {'state': 'completed'}, replace=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.read(path), {'state': 'running'})
        self.assertEqual(list(support._io_path(self.root).glob('.record-*')), [])

    def test_serialization_failure_cleans_temporary_without_publishing(self):
        path = self.root / 'status.json'
        with patch.object(support.json, 'dump', side_effect=ValueError('injected JSON failure')):
            with self.assertRaisesRegex(ValueError, 'injected JSON failure'):
                support.write_json_atomic(path, {'state': 'running'})
        self.assertFalse(support._io_path(path).exists())
        self.assertEqual(list(support._io_path(self.root).glob('.record-*')), [])

    def test_relative_write_returns_requested_spelling(self):
        previous = Path.cwd()
        try:
            os.chdir(support._io_path(self.root))
            path = Path('status.json')
            self.assertEqual(support.write_json_atomic(path, {'ok': True}), path)
            run = support.new_run_directory('run')
            self.assertEqual(run, self.root / 'run')
            self.assertEqual(support.new_run_directory('run', resume=True), run)
        finally:
            os.chdir(previous)

    @unittest.skipUnless(os.name == 'nt', 'Windows extended-path handling')
    def test_windows_temp_filename_can_cross_limit_before_target(self):
        # The public target itself is shorter than MAX_PATH, but the writer's
        # temporary filename exceeded it in the reported failure.
        directory = self.root / ('x' * (242 - len(str(self.root)) - 1))
        self.assertEqual(len(str(directory)), 242)
        run = support.new_run_directory(directory)
        target = run / 'status.json'
        self.assertLess(len(str(target)), 260)
        support.write_json_atomic(target, {'state': 'running'})
        support.write_json_atomic(target, {'state': 'completed'}, replace=True)
        self.assertEqual(self.read(target)['state'], 'completed')
        self.assertEqual(list(support._io_path(run).glob('.record-*')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows extended-path handling')
    def test_windows_deep_public_run_resume_and_failure_records(self):
        requested = self.root / ('a' * 90) / ('b' * 90) / ('c' * 90)
        self.assertGreater(len(str(requested)), 260)
        run = support.new_run_directory(requested)
        self.assertEqual(run, requested)
        self.assertFalse(str(run).startswith('\\\\?\\'))
        self.assertTrue(support._io_path(run / 'failures').is_dir())
        with self.assertRaises(FileExistsError):
            support.new_run_directory(requested)
        for name in ('run', 'status', 'coarse-cv', 'refit', 'completed'):
            path = run / (name + '.json')
            self.assertEqual(support.write_json_atomic(path, {'record': name}), path)
            self.assertEqual(self.read(path), {'record': name})
        self.assertEqual(support.new_run_directory(requested, resume=True), run)
        missing = run / 'not-a-run'
        with self.assertRaisesRegex(ValueError, 'existing search run'):
            support.new_run_directory(missing, resume=True)
        failure_path = run / 'failures' / 'failure.json'
        self.assertEqual(support.write_failure_json(failure_path, {'state': 'failed'}), failure_path)
        support.record_failure(run / 'failures', ValueError('example'), phase='cv_fit',
                               estimator=support.BaseEstimator(), training_data={'sha256': 'test'})
        failures = list(support._io_path(run / 'failures').glob('failure-*.json'))
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.read(failures[0])['phase'], 'cv_fit')
        self.assertEqual(list(support._io_path(run).rglob('.record-*')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows path syntax')
    def test_windows_drive_unc_and_already_extended_normalization(self):
        drive = self.root / 'name with spaces' / 'status.json'
        self.assertEqual(str(support._io_path(drive)), '\\\\?\\' + str(drive))
        unc = Path('\\\\server\\share\\folder\\status.json')
        self.assertEqual(str(support._io_path(unc)), '\\\\?\\UNC\\server\\share\\folder\\status.json')
        for path in (support._io_path(drive), support._io_path(unc)):
            self.assertEqual(support._io_path(path), path)
        # Only local paths are resolved; the UNC assertion above makes no
        # network request or assumption that a share is available.
        extended_run = support._io_path(self.root / 'extended-run')
        self.assertEqual(support.new_run_directory(extended_run), extended_run)
        self.assertEqual(support.new_run_directory(extended_run, resume=True), extended_run)

    @unittest.skipUnless(os.name == 'nt', 'Windows end-to-end long output paths')
    def test_windows_deep_adaptive_completed_resume_and_legacy_workflows(self):
        import adaptive_search
        import cv_checkpoint
        import tune_mnist

        parent = self.root / ('a' * 90) / ('b' * 90) / ('c' * 90)
        X = np.array([[-4.], [-3.], [-2.], [-1.], [1.], [2.], [3.], [4.]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        first, second = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
        folds = [(first, second), (second, first)]
        grid = {'lambd': [1.0], 'gamma': [1.0]}
        config = adaptive_search.AdaptiveSearchConfig(prefix='', refine=False)
        runtime = {'test_runtime': 'fixed'}

        # Isolate filesystem behavior from the independently tested search
        # policy and runtime inventory. Exercise real fold and final receipts.
        def next_stage(results, initial_grid, settings, state):
            if not results['params']:
                return {'state': {}, 'phase': 'coarse',
                        'candidates': [{'lambd': 1.0, 'gamma': 1.0}], 'review': None}
            return {'state': {}, 'phase': None, 'candidates': [],
                    'review': {'state': 'accepted', 'reasons': [],
                               'stopping_reason': 'fixed test policy',
                               'final_refit_performed': False, 'held_out_scored': False}}

        run = support.new_run_directory(parent / 'adaptive')
        PathWorkflowClassifier.fits = PathWorkflowClassifier.predictions = 0
        with patch.object(cv_checkpoint, 'runtime_fingerprint', return_value=runtime), \
                patch.object(adaptive_search, 'next_search_stage', side_effect=next_stage), \
                contextlib.redirect_stdout(io.StringIO()):
            result = adaptive_search.run_adaptive_search(
                PathWorkflowClassifier(), grid, X, y, X, y, folds, run, config=config)
            counts = (PathWorkflowClassifier.fits, PathWorkflowClassifier.predictions)
            resumed = adaptive_search.run_adaptive_search(
                PathWorkflowClassifier(), grid, X, y, X, y, folds, run,
                config=config, resume=True)
        self.assertEqual(result['run_dir'], run)
        self.assertTrue(resumed['resumed_completed'])
        self.assertEqual(counts, (3, 3))
        self.assertEqual((PathWorkflowClassifier.fits, PathWorkflowClassifier.predictions), counts)
        self.assertEqual(self.read(run / 'status.json')['state'], 'completed')
        for name in ('run', 'coarse-cv', 'accumulated-cv', 'selection', 'review', 'refit', 'completed'):
            self.assertTrue(support._io_path(run / (name + '.json')).is_file())

        legacy = support.new_run_directory(parent / 'legacy')
        with contextlib.redirect_stdout(io.StringIO()):
            result = tune_mnist.run_search_workflow(
                PathWorkflowClassifier(), grid, X, y, X, y, folds, legacy, verbose=0)
        self.assertEqual(result['run_dir'], legacy)
        self.assertEqual(self.read(legacy / 'status.json')['state'], 'completed')
        self.assertEqual(list(support._io_path(parent).rglob('.record-*')), [])

    @unittest.skipIf(os.name == 'nt', 'POSIX path behavior')
    def test_non_windows_io_preserves_relative_and_absolute_paths(self):
        for path in (Path('relative') / 'status.json', self.root / 'status.json'):
            self.assertEqual(support._io_path(path), path)


if __name__ == '__main__':
    unittest.main()
